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
from monai.transforms import Resized, Compose, LoadImaged, Orientationd, Spacing, Spacingd, EnsureTyped, EnsureChannelFirstd, AsDiscrete, CastToTyped, Resize, Resample, ResizeWithPadOrCrop
from nnunet.training.network_training.nnUNetTrainer import nnUNetTrainer
import nnunet
from collections import OrderedDict
from nnunet.training.data_augmentation.default_data_augmentation import default_3D_augmentation_params


from split_img_to_SA_LI_RI import saveDiffFrac

from monai.networks import nets 
import torch.nn as nn
import sys

import rootutils 
rootutils.setup_root("/home/shirshak/lightning-hydra-template", indicator=".project-root", pythonpath=True)


INPUT_PATH = Path("/home/shirshak/Pengwin_Submission_Portal/test/input")
OUTPUT_PATH = Path("/home/shirshak/Pengwin_Submission_Portal/test/output")
RESOURCE_PATH = Path("/home/shirshak/Pengwin_Submission_Portal/resources")


def change_direction(orig_image, direction_name = 'RAS'):
    new_img = sitk.DICOMOrient(orig_image, direction_name)
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

def return_one_with_max_probability(li_mask, sa_mask, ri_mask, li_prob, sa_prob, ri_prob):

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


def predict_li_sa_ri_files_from_trainer(trainer, frac_img):
    frac_array = sitk.GetArrayFromImage(frac_img)
    _, class_probabilities = trainer.predict_preprocessed_data_return_seg_and_softmax(np.expand_dims(frac_array, axis=0)) 
    # We needed to swap class probabilities first because of the swap axes between the original array and ct image as they read column wise and row wise, so that caused error......
    class_probabilities_swapped = class_probabilities.copy()
    class_probabilities_swapped[1] = class_probabilities[2]
    class_probabilities_swapped[2] = class_probabilities[1]
    predicted_segmentation = np.argmax(class_probabilities_swapped, axis=0)
    class_probabilities = class_probabilities_swapped

    # predicted_segmentation = np.argmax(class_probabilities, axis=0)

    # The output i.e predicted_segmentation gives 4 rectangular boxes on the endpoints, so to remove it we make the points at the end as 0
    predicted_segmentation[-20:, :, :] = 0     # We take a default parameter 20 after experimentation 
    predicted_segmentation[:, -20:, :] = 0
    predicted_segmentation[:, :, -20:] = 0
    predicted_segmentation[:20, :, :] = 0
    predicted_segmentation[:, :20, :] = 0
    predicted_segmentation[:, :, :20] = 0
    
    return predicted_segmentation, class_probabilities




def load_image_file_after_transform(*, location):
    val_transform = Compose([
        LoadImaged(keys=["image", "label"]),
        # CastToTyped(keys=["image", "label"], dtype=torch.int8),
        EnsureChannelFirstd(keys=["image","label"]),
        EnsureTyped(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"],pixdim=(1.0, 1.0, 1.0),mode=("bilinear", "nearest"),),
        Resized(keys=["image","label"],spatial_size=(128,128,128), mode=("area", "nearest")),
        # NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        # RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
        # RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
    ])
    
    train_images = glob(str(location / "*.mha"))
    if train_images:
        train_labels = train_images.copy()
        data_dicts = [{"image": image_name, "label": label_name} for image_name, label_name in zip(train_images, train_labels)]
        result = val_transform(data_dicts[0])
        return result, train_images[0]
    else:
        return None


def write_array_as_image_file(*, location, array):
    location.mkdir(parents=True, exist_ok=True)

    # You may need to change the suffix to .tiff to match the expected output
    suffix = ".mha"

    image = SimpleITK.GetImageFromArray(array)
    image.SetSpacing() #1,1,1
    image.SetDirection() # , 
    SimpleITK.WriteImage(
        image,
        location / f"output{suffix}",
        useCompression=True,
    )


def resample_volume(volume, new_spacing, interpolator = sitk.sitkNearestNeighbor): # Please choose nearest neighbor interpolator 
    # volume = sitk.ReadImage(volume_path, sitk.sitkFloat32) # read and cast to float32
    original_spacing = volume.GetSpacing()
    original_size = volume.GetSize()
    new_size = [int(round(osz*ospc/nspc)) for osz,ospc,nspc in zip(original_size, original_spacing, new_spacing)]
    return sitk.Resample(volume, new_size, sitk.Transform(), interpolator,
                         volume.GetOrigin(), new_spacing, volume.GetDirection(), 0,
                         volume.GetPixelID())

def resample_img(itk_image, out_spacing, is_label=False):
    
    # Resample images to 2mm spacing with SimpleITK
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()

    out_size = [
        int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
        int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
        int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2])))]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())

    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)

    return resample.Execute(itk_image)

def resize(img, new_size, interpolator):
    # img = sitk.ReadImage(img)
    dimension = img.GetDimension()

    # Physical image size corresponds to the largest physical size in the training set, or any other arbitrary size.
    reference_physical_size = np.zeros(dimension)

    reference_physical_size[:] = [(sz - 1) * spc if sz * spc > mx else mx for sz, spc, mx in
                                  zip(img.GetSize(), img.GetSpacing(), reference_physical_size)]

    # Create the reference image with a zero origin, identity direction cosine matrix and dimension
    reference_origin = np.zeros(dimension)
    reference_direction = np.identity(dimension).flatten()
    reference_size = new_size
    reference_spacing = [phys_sz / (sz - 1) for sz, phys_sz in zip(reference_size, reference_physical_size)]

    reference_image = sitk.Image(reference_size, img.GetPixelIDValue())
    reference_image.SetOrigin(reference_origin)
    reference_image.SetSpacing(reference_spacing)
    reference_image.SetDirection(reference_direction)

    # Always use the TransformContinuousIndexToPhysicalPoint to compute an indexed point's physical coordinates as
    # this takes into account size, spacing and direction cosines. For the vast majority of images the direction
    # cosines are the identity matrix, but when this isn't the case simply multiplying the central index by the
    # spacing will not yield the correct coordinates resulting in a long debugging session.
    reference_center = np.array(
        reference_image.TransformContinuousIndexToPhysicalPoint(np.array(reference_image.GetSize()) / 2.0))

    # Transform which maps from the reference_image to the current img with the translation mapping the image
    # origins to each other.
    transform = sitk.AffineTransform(dimension)
    transform.SetMatrix(img.GetDirection())
    transform.SetTranslation(np.array(img.GetOrigin()) - reference_origin)
    # Modify the transformation to align the centers of the original and reference image instead of their origins.
    centering_transform = sitk.TranslationTransform(dimension)
    img_center = np.array(img.TransformContinuousIndexToPhysicalPoint(np.array(img.GetSize()) / 2.0))
    centering_transform.SetOffset(np.array(transform.GetInverse().TransformPoint(img_center) - reference_center))

    # centered_transform = sitk.Transform(transform)
    # centered_transform.AddTransform(centering_transform)

    centered_transform = sitk.CompositeTransform([transform, centering_transform])

    # Using the linear interpolator as these are intensity images, if there is a need to resample a ground truth
    # segmentation then the segmentation image should be resampled using the NearestNeighbor interpolator so that
    # no new labels are introduced.

    return sitk.Resample(img, reference_image, centered_transform, interpolator, 0.0)



def run():
    # WE DON'T NEED TO CHANGE DIRN OF IMG HERE becoz monai transforms will do it.
    print("Just Started")
    sys.stdout.write('Just started \n')

    # Loading Model 1 
    model = nets.UNet(spatial_dims=3, in_channels=1, out_channels=4, channels=[16,32,64], strides=[2,2])
    model_file_state_dict = torch.load(join(RESOURCE_PATH, 'best.ckpt'))['state_dict']
    pretrained_dict = {key.replace("net.", ""): value for key, value in model_file_state_dict.items()}
    model.load_state_dict(pretrained_dict)

    # Loading Model 2
    pkl = join(RESOURCE_PATH, 'model_best.model.pkl')
    checkpoint = pkl[:-4]
    train = False
    trainer = restore_model(pkl, checkpoint, train)
    # We also have to put value of data_aug_params from nnunet/training/data_augumentation/default_data_augumentation.py, and since our model is 3d full res model 
    trainer.data_aug_params = default_3D_augmentation_params

    # we take img_shape, and original_image for retrieving the shape and sizes of the overall image after prediction from first model. 
    pelvic_fracture_ct, original_image_name = load_image_file_after_transform(location=INPUT_PATH)

    if pelvic_fracture_ct is not None:
        # print(pelvic_fracture_ct["image"].shape) #torch.Size([1, 128, 128, 128])
        logits = model.forward(pelvic_fracture_ct["image"].unsqueeze(0)) # as shape needed by model is 1,1,128,128,128 bs,ch,h,w,d 
        softmax_logits = nn.Softmax(dim=1)(logits)
        predicted_segmentation = torch.argmax(softmax_logits, 1)
        
        # print(predicted_segmentation.shape) # torch.Size([1, 128, 128, 128])
        # print(pelvic_fracture_ct["image"].shape) # torch.Size([1, 128, 128, 128])

        frac_LeftIliac_img, frac_sacrum_img, frac_RightIliac_img = saveDiffFrac(sitk.GetImageFromArray(pelvic_fracture_ct["image"][0]), sitk.GetImageFromArray(predicted_segmentation[0]))

        sys.stdout.write('<----------Anatomical Model baseline unet completed---------------------> \n')
        # print('<----------Anatomical Model baseline unet completed--------------------->')

        predicted_frac_LeftIliac_img, class_prob_frac_LeftIliac_img = predict_li_sa_ri_files_from_trainer(trainer, frac_LeftIliac_img)
        predicted_frac_sacrum_img, class_prob_frac_sacrum_img = predict_li_sa_ri_files_from_trainer(trainer, frac_sacrum_img)
        predicted_frac_RightIliac_img, class_prob_frac_RightIliac_img = predict_li_sa_ri_files_from_trainer(trainer, frac_RightIliac_img)
        
        # sitk.WriteImage(sitk.GetImageFromArray(predicted_frac_LeftIliac_img), join(OUTPUT_PATH, "left_iliac.nii.gz"))
        # sitk.WriteImage(sitk.GetImageFromArray(predicted_frac_sacrum_img), join(OUTPUT_PATH, "sacrum_iliac.nii.gz"))
        # sitk.WriteImage(sitk.GetImageFromArray(predicted_frac_RightIliac_img), join(OUTPUT_PATH, "right_iliac.nii.gz"))

        sys.stdout.write('<----------Fracture Segmentation Model completed---------------------> \n')
        # print('<----------Fracture Segmentation Model completed--------------------->')

        min_valid_object_size = 500

        mask_preprocessed_arr = separate_labels_for_non_connected_splitted_fragments(predicted_frac_LeftIliac_img, for_which_classes=None, volume_per_voxel=float(np.prod(sitk.GetImageFromArray(predicted_frac_LeftIliac_img).GetSpacing(), dtype=np.float64)), minimum_valid_object_size=min_valid_object_size)
        li_mask = get_preds(mask_preprocessed_arr, 10)

        mask_preprocessed_arr = separate_labels_for_non_connected_splitted_fragments(predicted_frac_sacrum_img, for_which_classes=None, volume_per_voxel=float(np.prod(sitk.GetImageFromArray(predicted_frac_sacrum_img).GetSpacing(), dtype=np.float64)), minimum_valid_object_size=min_valid_object_size)
        sa_mask = get_preds(mask_preprocessed_arr, 0)

        mask_preprocessed_arr = separate_labels_for_non_connected_splitted_fragments(predicted_frac_RightIliac_img, for_which_classes=None, volume_per_voxel=float(np.prod(sitk.GetImageFromArray(predicted_frac_RightIliac_img).GetSpacing(), dtype=np.float64)), minimum_valid_object_size=min_valid_object_size)
        ri_mask = get_preds(mask_preprocessed_arr, 20)

        overall_mask = return_one_with_max_probability(li_mask, sa_mask, ri_mask, li_prob = class_prob_frac_LeftIliac_img, sa_prob = class_prob_frac_sacrum_img, ri_prob=class_prob_frac_RightIliac_img)

        overall_mask = overall_mask.astype(np.int8)
        print(f'Model output:{overall_mask.shape}')
        print(np.unique(overall_mask, return_counts=True))


        # model output completed ----------------
        orig_reader = sitk.ImageFileReader()
        orig_reader.SetFileName(original_image_name)
        orig_reader.ReadImageInformation()
        
        original_size = np.array(orig_reader.GetSize())
        original_size[0],original_size[-1] = original_size[-1],original_size[0]

        # resize to original shape
        overall_mask_resized = Resize(spatial_size=list(original_size),mode='nearest')(np.expand_dims(overall_mask,axis=0))[0] # undo expand_dims

        # convert to mha
        overall_mask_resized_sitk = sitk.GetImageFromArray(overall_mask_resized)
        overall_mask_resized_sitk.SetSpacing(orig_reader.GetSpacing())
        overall_mask_resized_sitk.SetDirection((-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0))
        # convert to original orientation
        original_orientation = sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(orig_reader.GetDirection())
        overall_mask_resized_reoriented_sitk = sitk.DICOMOrient(overall_mask_resized_sitk,original_orientation)
        sitk.WriteImage(overall_mask_resized_reoriented_sitk, './test/output/images/output.mha',useCompression=True)

        sys.exit()


        
        new_shape = np.asarray(sitk.GetArrayFromImage(sitk.ReadImage(original_image_name)).shape) * np.asarray(sitk.ReadImage(original_image_name).GetSpacing())
        # as int and nd.int64 were different so, resize function gave errors.... so we convert nd array nd.int64 to int
        new_shape = [int(x) for x in np.round(new_shape)]
        # Converting this to tuple as our resize function takes tuple and not array
        new_shape = tuple(new_shape)


        overall_mask_resized = Resize(spatial_size=new_shape, mode="nearest")(torch.from_numpy(overall_mask).unsqueeze(dim=0))

        overall_mask_resized_img = sitk.GetImageFromArray(overall_mask_resized.squeeze(dim=0))

        print(overall_mask_resized_img.GetSize())
        print(overall_mask_resized_img.GetSpacing())

        # overall_mask_resized_img = sitk.GetImageFromArray(np.swapaxes(sitk.GetArrayFromImage(overall_mask_resized_img), 0, 2))
        overall_mask_resized_img = resample_volume(overall_mask_resized_img, new_spacing=sitk.ReadImage(original_image_name).GetSpacing())
        
        print(overall_mask_resized_img.GetSize())
        print(overall_mask_resized_img.GetSpacing())

        overall_mask_resized_img = resize(overall_mask_resized_img, sitk.GetArrayFromImage(sitk.ReadImage(original_image_name)).shape, interpolator=sitk.sitkNearestNeighbor)
        print(overall_mask_resized_img.GetSize())
        print(overall_mask_resized_img.GetSpacing())

        overall_mask_resized_img = change_direction(overall_mask_resized_img, direction_name = sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(sitk.ReadImage(original_image_name).GetDirection()))
        print(overall_mask_resized_img.GetSize())
        print(overall_mask_resized_img.GetSpacing())

        print("-"*80)

        print(overall_mask_resized_img.GetSize())
        print(overall_mask_resized_img.GetSpacing())
        print(sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(overall_mask_resized_img.GetDirection()))
        
        # change_direction(orig_image, direction_name = 'RAS')
        print(sitk.ReadImage(original_image_name).GetSize())
        print(sitk.GetArrayFromImage(sitk.ReadImage(original_image_name)).shape)
        print(sitk.ReadImage(original_image_name).GetSpacing())
        print(sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(sitk.ReadImage(original_image_name).GetDirection()))



        Path(join(OUTPUT_PATH, 'images/pelvic-fracture-ct-segmentation')).mkdir(parents=True, exist_ok=True)
        write_array_as_image_file(location=OUTPUT_PATH / "images/pelvic-fracture-ct-segmentation", array=sitk.GetArrayFromImage(overall_mask_resized_img),)
    else:
        print("the image is none")
    return 0



if __name__ == "__main__":
    # import gdown 
    # gdown.download("https://drive.google.com/uc?id=1TDlfk8tGhMRIvk86nG8yspna2ZZ1-Lf0", join(RESOURCE_PATH, 'model_best.model'))
    raise SystemExit(run())


# PYTHONPATH=/path/to/Project python script.py
# PYTHONPATH=/home/shirshak/Anatomical_Segmentation_Frac-Seg-Net python3 /home/shirshak/Just-nnUNet-not-Overridden-with-FracSegNet-in-venv/PENGWIN-example-algorithm/PENGWIN-challenge-packages/preliminary-development-phase-ct/inference.py

