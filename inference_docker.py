"""
The following is a simple example algorithm.

It is meant to run within a container.

To run it locally, you can call the following bash script:

  ./test_run.sh

This will start the inference and reads from ./test/input and outputs to ./test/output

To save the container and prep it for upload to Grand-Challenge.org you can call:

  ./save.sh

Any container that shows the same behavior will do, this is purely an example of how one COULD do it.

Happy programming!
"""
from pathlib import Path

from glob import glob
import SimpleITK as sitk 
import SimpleITK
from split_img_to_SA_LI_RI import saveDiffFrac
from batchgenerators.utilities.file_and_folder_operations import load_pickle

from nnunet.inference.predict_simple import predict_from_folder

from nnunet.training.model_restore import recursive_find_python_class #, load_model_and_checkpoint_files, restore_model
import numpy as np

# from nnUNet.nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
# from nnUNet.nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from batchgenerators.utilities.file_and_folder_operations import join
import torch 
import os 
from scipy.ndimage import label
from torch.utils.data import Dataset, DataLoader
from monai.transforms import Resized, Compose, LoadImaged, Orientationd, Spacingd, EnsureTyped, EnsureChannelFirstd, AsDiscrete
from nnunet.training.network_training.nnUNetTrainer import nnUNetTrainer
import nnunet
from collections import OrderedDict
from nnunet.training.data_augmentation.default_data_augmentation import default_3D_augmentation_params


from split_img_to_SA_LI_RI import saveDiffFrac

from monai.networks import nets 
import torch.nn as nn

INPUT_PATH = Path("./input/images/pelvic-fracture-ct/")
OUTPUT_PATH = Path("./output")
RESOURCE_PATH = Path("./resources")

def change_direction(orig_image):
    new_img = sitk.DICOMOrient(orig_image, 'RAS')
    return new_img


def separate_labels_for_non_connected_splitted_fragments(mask_arr: np.ndarray, for_which_classes: list, volume_per_voxel: float,
                                                   minimum_valid_object_size: int = None) -> np.ndarray:
    """
    gives separate label for non connected components other than main fragment in an mask array.
    :param image:
    :param for_which_classes: can be None. Should be list of int.
    :param minimum_valid_object_size: Only objects larger than minimum_valid_object_size will be removed.
    :return:
    """
    if for_which_classes is None:
        for_which_classes = np.unique(mask_arr)
        for_which_classes = for_which_classes[for_which_classes > 1] # Not taking background (0) and largest class (1)

    assert 0 not in for_which_classes, "background scannot be incorporated"
    assert 1 not in for_which_classes, "largest class, class 1  couldnot be incorporated, only small fragments can be incorporated"


    for c in for_which_classes: # for_which_classes = [2]
        mask = np.zeros_like(mask_arr, dtype=bool)
        mask = mask_arr == c


        # get labelmap and number of objects
        lmap, num_objects = label(mask.astype(int))

        # collect object sizes
        object_sizes = {}
        for object_id in range(1, num_objects + 1):
            object_sizes[object_id] = (lmap == object_id).sum() * volume_per_voxel

        # Removing smaller objects which is lesser than threshold
        if num_objects > 0:
            for object_id in range(1, num_objects + 1):
                    if minimum_valid_object_size is not None:
                        if object_sizes[object_id] < minimum_valid_object_size:
                            mask_arr[(lmap == object_id) & mask] = 0
                            del object_sizes[object_id]

        num_objects = len(object_sizes)

        if num_objects > 0:
            mask_value = 2 # Mask value for background is already 0, for largest bone is already 1, so starting from 2 for sub-fragments, and taking 2 for largest sub fragment bone...
            for _ in range(num_objects):
                # print('printing object sizesssssssss')
                # print(len(object_sizes))
                if len(object_sizes) > 1:
                    maximum_size = max(object_sizes.values()) 
                else:
                # For final component, comparing it with minimum valid object if it is not None. If it is none comparing it with 0
                    maximum_size = max(list(object_sizes.values())[0], [minimum_valid_object_size if minimum_valid_object_size is not None else 0])
                    
                for object_id in list(object_sizes.keys()):
                    # print(object_sizes[object_id])
                    # print(maximum_size)
                    if object_sizes[object_id] == maximum_size:
                        # mark that as label mask_value(2)
                        mask_arr[(lmap == object_id) & mask] = mask_value
                        # remove that object_sizes[object_id]
                        del object_sizes[object_id]
                        # print(f"Remove and deleted largest component {object_id}")
                        break
                    else:
                        continue 
                mask_value += 1   
    return mask_arr

def get_preds(mask_preprocessed_arr, base_value):
    mask = mask_preprocessed_arr.copy()
    mask[mask_preprocessed_arr == 1] = base_value + 1
    mask[mask_preprocessed_arr == 2] = base_value + 2
    mask[mask_preprocessed_arr == 3] = base_value + 3
    mask[mask_preprocessed_arr == 4] = base_value + 4
    mask[mask_preprocessed_arr == 5] = base_value + 5
    mask[mask_preprocessed_arr == 6] = base_value + 6
    return mask

def return_one_with_max_probability(li_mask, sa_mask, ri_mask, location, str_img_number):

    li_prob = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(location, f'{str_img_number}_LI_pred_prob.nii.gz')))
    sa_prob = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(location, f'{str_img_number}_SA_pred_prob.nii.gz')))
    ri_prob = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(location, f'{str_img_number}_RI_pred_prob.nii.gz')))

    # print(li_prob.shape) # (3, 128, 128, 128) #SInce there is background, class 1, class 2....
    # print(sa_prob.shape) # (3, 128, 128, 128)
    # print(ri_prob.shape) # (3, 128, 128, 128)

    li_prob = li_prob.max(axis=0)
    sa_prob = sa_prob.max(axis=0)
    ri_prob = ri_prob.max(axis=0)

    assert ri_mask.shape == li_mask.shape ==sa_mask.shape == li_prob.shape == ri_prob.shape == sa_prob.shape
    overall_mask = np.zeros_like(li_mask)

    for i in range(ri_mask.shape[0]):
        for j in range(ri_mask.shape[1]):
            for k in range(ri_mask.shape[2]):
                my_dict = {}
                # print(li_mask[i][j][k])
                if li_mask[i][j][k] != 0:
                    my_dict[li_mask[i][j][k]] = li_prob[i][j][k]

                if ri_mask[i][j][k] != 0:
                    my_dict[ri_mask[i][j][k]] = ri_prob[i][j][k]

                if sa_mask[i][j][k] != 0:
                    my_dict[sa_mask[i][j][k]] = sa_prob[i][j][k]

                # This even works if there is only one element in dict
                if len(my_dict) != 0 :
                    overall_mask[i][j][k] = max(my_dict, key=my_dict.get)
                else:
                    overall_mask[i][j][k] = 0
    return overall_mask 



def get_overall_segmentation_of_one_img(str_img_number, location):
    '''
    str_img_number: example -----> 049, 001, etc.
    '''
    mask_name_LI = os.path.join(location, f'{str_img_number}_LI_pred.nii.gz')
    mask_name_SA = os.path.join(location, f'{str_img_number}_SA_pred.nii.gz')
    mask_name_RI = os.path.join(location, f'{str_img_number}_RI_pred.nii.gz')

    min_valid_object_size = 500

    mask_preprocessed_arr = separate_labels_for_non_connected_splitted_fragments(sitk.GetArrayFromImage(sitk.ReadImage(mask_name_LI)), for_which_classes=None, volume_per_voxel=float(np.prod(sitk.ReadImage(mask_name_LI).GetSpacing(), dtype=np.float64)), minimum_valid_object_size=min_valid_object_size)
    li_mask = get_preds(mask_preprocessed_arr, 10)

    mask_preprocessed_arr = separate_labels_for_non_connected_splitted_fragments(sitk.GetArrayFromImage(sitk.ReadImage(mask_name_SA)), for_which_classes=None, volume_per_voxel=float(np.prod(sitk.ReadImage(mask_name_SA).GetSpacing(), dtype=np.float64)), minimum_valid_object_size=min_valid_object_size)
    sa_mask = get_preds(mask_preprocessed_arr, 0)

    mask_preprocessed_arr = separate_labels_for_non_connected_splitted_fragments(sitk.GetArrayFromImage(sitk.ReadImage(mask_name_RI)), for_which_classes=None, volume_per_voxel=float(np.prod(sitk.ReadImage(mask_name_RI).GetSpacing(), dtype=np.float64)), minimum_valid_object_size=min_valid_object_size)
    ri_mask = get_preds(mask_preprocessed_arr, 20)

    print(np.unique(li_mask, return_counts=True))
    print(np.unique(sa_mask, return_counts=True))
    print(np.unique(ri_mask, return_counts=True))

    overall_mask = return_one_with_max_probability(li_mask, sa_mask, ri_mask, location, str_img_number)

    print(np.unique(overall_mask, return_counts=True))
    return overall_mask



# taken from nnunet.training.model_restore.py -> restore_model -> trainer.load_checkpoint() 
def restore_model(pkl_file, checkpoint=None, train=False, fp16=None):
    """
    This is a utility function to load any nnUNet trainer from a pkl. It will recursively search
    nnunet.trainig.network_training for the file that contains the trainer and instantiate it with the arguments saved in the pkl file. If checkpoint
    is specified, it will furthermore load the checkpoint file in train/test mode (as specified by train).
    The pkl file required here is the one that will be saved automatically when calling nnUNetTrainer.save_checkpoint.
    :param pkl_file:
    :param checkpoint:
    :param train:
    :param fp16: if None then we take no action. If True/False we overwrite what the model has in its init
    :return:
    """
    info = load_pickle(pkl_file)
    init = info['init']
    name = info['name']
    search_in = join(nnunet.__path__[0], "training", "network_training")
    tr = recursive_find_python_class([search_in], name, current_module="nnunet.training.network_training")


    if tr is None:
        raise RuntimeError("Could not find the model trainer specified in checkpoint in nnunet.trainig.network_training. If it "
                           "is not located there, please move it or change the code of restore_model. Your model "
                           "trainer can be located in any directory within nnunet.trainig.network_training (search is recursive)."
                           "\nDebug info: \ncheckpoint file: %s\nName of trainer: %s " % (checkpoint, name))
    assert issubclass(tr, nnUNetTrainer), "The network trainer was found but is not a subclass of nnUNetTrainer. " \
                                          "Please make it so!"

    # this is now deprecated
    """if len(init) == 7:
        print("warning: this model seems to have been saved with a previous version of nnUNet. Attempting to load it "
              "anyways. Expect the unexpected.")
        print("manually editing init args...")
        init = [init[i] for i in range(len(init)) if i != 2]"""

    # ToDo Fabian make saves use kwargs, please...

    trainer = tr(*init)

    # We can hack fp16 overwriting into the trainer without changing the init arguments because nothing happens with
    # fp16 in the init, it just saves it to a member variable
    if fp16 is not None:
        trainer.fp16 = fp16

    # taken from nnunet.training.model_restore.py -> restore_model -> trainer.load_checkpoint() 
    trainer.process_plans(info['plans'])
    saved_model = torch.load(checkpoint, map_location=torch.device('cpu'))

    # Now this(upto return statement) is taken from nnunet.training.network_trainer.network_trainer.py -> load_checkpoint -> trainer.load_checkpoint_ram function
    new_state_dict = OrderedDict()

    # curr_state_dict_keys = list(trainer.network.state_dict().keys())
    # if state dict comes form nn.DataParallel but we use non-parallel model here then the state dict keys do not
    # match. Use heuristic to make it match
    for k, value in saved_model['state_dict'].items():
        key = k
        if key.startswith('module.'):
            key = key[7:]
        new_state_dict[key] = value

    trainer.initialize_network()
    trainer.network.load_state_dict(new_state_dict)

    return trainer


def run():
    # Read the input
    # pelvic_facture_ct = load_image_file_as_array(
    #     location=INPUT_PATH / "001.mha",
    # )
    # # Process the inputs: any way you'd like
    # _show_torch_cuda_info()


    # FOR Anatomical Model USING UNet baseline 
    # WE DON'T NEED TO CHANGE DIRN OF IMG HERE becoz monai transforms will do it.

    os.makedirs(join(OUTPUT_PATH, 'output_data_anatomical'), exist_ok=True)
    output_dir_anatomical = join(OUTPUT_PATH, 'output_data_anatomical')

    class HelperDataset(Dataset):
        def __init__(self, file_names, transform):
            self.file_names = file_names
            self.transform = transform

        def __getitem__(self, index):
            file_names = self.file_names[index]
            dataset = self.transform(file_names) 
            return dataset
        
        def __len__(self):
            return len(self.file_names)
        
    class UNet(nets.UNet):
        def __init__(self,spatial_dims, in_channels, out_channels, 
                    channels,strides):
            super().__init__(spatial_dims, in_channels, out_channels, channels, strides)

        def forward(self, **kwargs) -> torch.Tensor:
            image = kwargs["image"]
            return super().forward(image)
        
    val_transform = Compose([
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image","label"]),
            EnsureTyped(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(keys=["image", "label"],pixdim=(1.0, 1.0, 1.0),mode=("bilinear", "nearest"),),
            Resized(keys=["image","label"],spatial_size=(128,128,128), mode=("area", "nearest")),
            # NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            # RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
            # RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
        ])
    
    train_images = sorted(glob(join(INPUT_PATH, '*.mha')))
    train_labels = train_images.copy() #sorted(glob(join(INPUT_PATH, '*_label.mha')))
    data_dicts = [{"image": image_name, "label": label_name} for image_name, label_name in zip(train_images, train_labels)]

    my_dataset = HelperDataset(data_dicts, val_transform)

    dataloader = DataLoader(dataset=my_dataset, batch_size=1)

    model = UNet(spatial_dims=3, in_channels=1, out_channels=4, channels=[16,32,64], strides=[2,2])

    model_file_state_dict = torch.load(join(RESOURCE_PATH, 'best.ckpt'))['state_dict']
    
    pretrained_dict = {key.replace("net.", ""): value for key, value in model_file_state_dict.items()}
    model.load_state_dict(pretrained_dict)

    for data in dataloader:
        logits = model.forward(**data)
        softmax_logits = nn.Softmax(dim=1)(logits)
        predicted_segmentation = torch.argmax(softmax_logits, 1)
        predicted_segmentation = predicted_segmentation.squeeze(dim=0)
        only_name = os.path.split(data['image_meta_dict']['filename_or_obj'][0])[1].split('.mha')[0]
        print(np.unique(predicted_segmentation))

        # # We write label just for checking what label is after passing to transforms of monai 
        # sitk.WriteImage(sitk.GetImageFromArray(data["label"]), join(output_dir_anatomical, only_name+'_label.nii.gz'))
        
        # We write image as well because we have resized the image so, while passing to saveDiffFrac img & label should be same size
        sitk.WriteImage(sitk.GetImageFromArray(data["image"]), join(output_dir_anatomical, only_name+'_image.nii.gz'))
        sitk.WriteImage(sitk.GetImageFromArray(predicted_segmentation), join(output_dir_anatomical, only_name + '_pred.nii.gz'))

        os.makedirs(join(output_dir_anatomical, 'splitted_fragments'), exist_ok=True)
        saveDiffFrac(join(output_dir_anatomical, only_name + '_image.nii.gz'), join(output_dir_anatomical, only_name + '_pred.nii.gz'), join(output_dir_anatomical,'splitted_fragments'))

    print('<----------Anatomical Model baseline unet completed--------------------->')


    # # For Frac_Seg Model 

    output_dir_anatomical = join(OUTPUT_PATH, 'output_data_anatomical')
    os.makedirs(join(OUTPUT_PATH, 'output_data_fracsegnet'), exist_ok=True)
    output_dir_fracsegnet = join(OUTPUT_PATH, 'output_data_fracsegnet')
    
    pkl = join(RESOURCE_PATH, 'model_best.model.pkl')
    checkpoint = pkl[:-4]
    train = False

    trainer = restore_model(pkl, checkpoint, train)

    preprocessed_img_folder =  join(output_dir_anatomical, 'splitted_fragments', 'nifty_preprocessed_into_fragments',  'images/*.nii.gz')

    for img_name in sorted(glob(preprocessed_img_folder)):

        pelvic_facture_ct = SimpleITK.GetArrayFromImage(SimpleITK.ReadImage(img_name))

        split_name = os.path.split(img_name)
        only_name = split_name[1].split('_image.nii.gz', 2)[0]

        print(only_name + "_image.nii.gz")

        # print(np.expand_dims(pelvic_facture_ct, axis=0).shape) # extra dimension for channel [1,416,512,512]

        # We also have to put value of data_aug_params from nnunet/training/data_augumentation/default_data_augumentation.py, and since our model is 3d full res model 
        trainer.data_aug_params = default_3D_augmentation_params
        # trainer.data_aug_params["do_mirror"] = (0, 1, 2)
        # this will return tuple of predicted_segmentation, class_probabilities so we take only predicted segmentation
        _, class_probabilities = trainer.predict_preprocessed_data_return_seg_and_softmax(np.expand_dims(pelvic_facture_ct, axis=0)) 

        
        # I don't know but the output of model2 labels 1 for smaller bone and 2 for larger bone everytime for the output coming from the
        # 1st model when feeded to input of 2nd model it gives that. BYou might say you have to do preprocessing first, but after preprocessing 
        # the output of 2nd model gives worse output, hence we don't do preprocess as before passing to anatomical(1st) model we do necessary preprocessing.
        # Hence Now, we want to replace model's prediction of  class 1 as class 2 and vice versa. As it gave that error everytime, after the output of anatomical model. 
        # But on good data the model performs nicely. Hence, we swap class 1 as 2 and class 2 as 1 
        
        # print(class_probabilities.shape) # (3, 128, 128, 128)
        class_probabilities_swapped = class_probabilities.copy()
        class_probabilities_swapped[1] = class_probabilities[2]
        class_probabilities_swapped[2] = class_probabilities[1]


        predicted_segmentation = np.argmax(class_probabilities_swapped, axis=0)
        class_probabilities = class_probabilities_swapped

        print(np.unique(predicted_segmentation,return_counts=True))

        # print(class_probabilities.shape) # (3, 61, 60, 110)
        
        
        # The output i.e predicted_segmentation gives 4 rectangular boxes on the endpoints, so to remove it we make the points at the end as 0

        predicted_segmentation[-20:, :, :] = 0     # We take a default parameter 20 after experimentation 
        predicted_segmentation[:, -20:, :] = 0
        predicted_segmentation[:, :, -20:] = 0
        predicted_segmentation[:20, :, :] = 0
        predicted_segmentation[:, :20, :] = 0
        predicted_segmentation[:, :, :20] = 0

        sitk.WriteImage(sitk.GetImageFromArray(predicted_segmentation), join(output_dir_fracsegnet, only_name + "_pred.nii.gz"))
        sitk.WriteImage(sitk.GetImageFromArray(class_probabilities), join(output_dir_fracsegnet, only_name + "_pred_prob.nii.gz"))

    print('<----------Fracture Segmentation Model completed--------------------->')

    # img_number = ['001', '002']
    # Same operation like line 304 but we automate it.
    img_number = set()
    for img in sorted(glob(join(output_dir_fracsegnet, '*_pred.nii.gz'))):
        img_number.add(os.path.split(img)[1][:3])
    img_number = list(img_number)

    print(f"Number of image is : {img_number}")

    os.makedirs(join(OUTPUT_PATH, 'images/pelvic-fracture-ct-segmentation'), exist_ok=True)
    final_output_folder = join(OUTPUT_PATH, 'images/pelvic-fracture-ct-segmentation')

    for img_num in img_number:
        # print(img_num)
        overall_mask_of_whole_img = get_overall_segmentation_of_one_img(img_num, output_dir_fracsegnet)
        sitk.WriteImage(sitk.GetImageFromArray(overall_mask_of_whole_img), join(final_output_folder, f'{img_num}_overall_pred.mha'))

    # # Save your output
    # # write_array_as_image_file(
    # #     location=OUTPUT_PATH / "images/pelvic-fracture-ct-segmentation",
    # #     array=pelvic_fracture_segmentation,
    # # )
    return 0


def load_image_file_as_array(*, location):
    # Use SimpleITK to read a file
    input_files = glob(str(location / "*.mha"))
    result = SimpleITK.ReadImage(input_files[0])

    # Convert it to a Numpy array
    return SimpleITK.GetArrayFromImage(result)


def write_array_as_image_file(*, location, array):
    location.mkdir(parents=True, exist_ok=True)

    # You may need to change the suffix to .tiff to match the expected output
    suffix = ".mha"

    image = SimpleITK.GetImageFromArray(array)
    SimpleITK.WriteImage(
        image,
        location / f"output{suffix}",
        useCompression=True,
    )


def _show_torch_cuda_info():
    print("=+=" * 10)
    print("Collecting Torch CUDA information")
    print(f"Torch CUDA is available: {(available := torch.cuda.is_available())}")
    if available:
        print(f"\tnumber of devices: {torch.cuda.device_count()}")
        print(f"\tcurrent device: { (current_device := torch.cuda.current_device())}")
        print(f"\tproperties: {torch.cuda.get_device_properties(current_device)}")
    print("=+=" * 10)


if __name__ == "__main__":
    raise SystemExit(run())


# PYTHONPATH=/path/to/Project python script.py
# PYTHONPATH=/home/shirshak/Anatomical_Segmentation_Frac-Seg-Net python3 /home/shirshak/Just-nnUNet-not-Overridden-with-FracSegNet-in-venv/PENGWIN-example-algorithm/PENGWIN-challenge-packages/preliminary-development-phase-ct/inference.py
