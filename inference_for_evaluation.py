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
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
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
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO 

from monai.networks import nets 
import torch.nn as nn
import sys
from evaluation_CT_core import load_image_file, load_gt_label_and_spacing, evaluate_3d_single_case

import rootutils 
rootutils.setup_root("/home/shirshak/lightning-hydra-template", indicator=".project-root", pythonpath=True)

from evaluation_CT_core import evaluate_3d_single_case



INPUT_PATH = Path("/home/shirshak/Pengwin_Submission_Portal/test/input")
LABEL_PATH = Path("/home/shirshak/Pengwin_Submission_Portal/test/label")
OUTPUT_PATH = Path("/home/shirshak/Pengwin_Submission_Portal/test/output")
RESOURCE_PATH = Path("/home/shirshak/Pengwin_Submission_Portal/resources")


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

def return_one_with_max_probability(li_mask, sa_mask, ri_mask, li_prob, sa_prob, ri_prob):

    # print(li_prob.shape) # (3, 128, 128, 128) #SInce there is background, class 1, class 2....
    # print(sa_prob.shape) # (3, 128, 128, 128)
    # print(ri_prob.shape) # (3, 128, 128, 128)

    li_prob = li_prob.max(axis=0)
    sa_prob = sa_prob.max(axis=0)
    ri_prob = ri_prob.max(axis=0)

    assert ri_mask.shape == li_mask.shape ==sa_mask.shape == li_prob.shape == ri_prob.shape == sa_prob.shape
    overall_mask = np.zeros_like(li_mask)

    overall_mask = li_mask + ri_mask + sa_mask      # for quick submission
    
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


def predict_li_sa_ri_files_from_trainer(trainer, frac_array):
    predicted_segmentation, class_probabilities = trainer.predict_preprocessed_data_return_seg_and_softmax(np.expand_dims(frac_array, axis=0)) 
    # class_probabilities_swapped = class_probabilities.copy()
    # class_probabilities_swapped[1] = class_probabilities[2]
    # class_probabilities_swapped[2] = class_probabilities[1]
    # predicted_segmentation = np.argmax(class_probabilities_swapped, axis=0)
    # class_probabilities = class_probabilities_swapped

    # The output i.e predicted_segmentation gives 4 rectangular boxes on the endpoints, so to remove it we make the points at the end as 0
    predicted_segmentation[-20:, :, :] = 0     # We take a default parameter 20 after experimentation 
    predicted_segmentation[:, -20:, :] = 0
    predicted_segmentation[:, :, -20:] = 0
    predicted_segmentation[:20, :, :] = 0
    predicted_segmentation[:, :20, :] = 0
    predicted_segmentation[:, :, :20] = 0
    
    return predicted_segmentation, class_probabilities


def run():
    # WE DON'T NEED TO CHANGE DIRN OF IMG HERE becoz monai transforms will do it.
    print("Just Started")
    sys.stdout.write('Just started \n')

    # Loading Model 1 
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=False,#True,
        use_mirroring=False, #True,
        perform_everything_on_device=True,
        device=torch.device('cuda', 0),
        verbose=True,
        verbose_preprocessing=True,
        allow_tqdm=True
    )

    predictor.initialize_from_trained_model_folder(
        join('/home/shirshak/Pengwin_Submission_Portal/resources', 'Dataset604_CT_PelvicFrac150/nnUNetTrainer__nnUNetPlans__3d_lowres'),
        use_folds=(4,),
        checkpoint_name='checkpoint_best.pth',
    )

    # Loading Model 2
    pkl = join(RESOURCE_PATH, 'model_best.model.pkl')
    checkpoint = pkl[:-4]
    train = False
    trainer = restore_model(pkl, checkpoint, train)
    # We also have to put value of data_aug_params from nnunet/training/data_augumentation/default_data_augumentation.py, and since our model is 3d full res model 
    trainer.data_aug_params = default_3D_augmentation_params

    # we take img_shape, and original_image for retrieving the shape and sizes of the overall image after prediction from first model. 

    for img_index, (train_img, train_label) in enumerate(zip(sorted(glob(str(INPUT_PATH / "images/pelvic-fracture-ct/*.mha"))), sorted(glob(str(LABEL_PATH / "*.mha")))), 1):
        original_img = sitk.ReadImage(train_img)
        original_arr, props = SimpleITKIO().read_images([train_img])

        predicted_segmentation, class_probabilities = predictor.predict_single_npy_array(original_arr, props, None, None, True)
        
        print(np.unique(predicted_segmentation, return_counts=True))
        print(predicted_segmentation.shape)
    
        frac_LeftIliac_arr, frac_sacrum_arr, frac_RightIliac_arr = saveDiffFrac(original_arr, predicted_segmentation)

        sys.stdout.write('<----------Anatomical Model baseline unet completed---------------------> \n')
        # print('<----------Anatomical Model baseline unet completed--------------------->')

        predicted_frac_LeftIliac_arr, class_prob_frac_LeftIliac_arr = predict_li_sa_ri_files_from_trainer(trainer, frac_LeftIliac_arr)
        predicted_frac_sacrum_arr, class_prob_frac_sacrum_arr = predict_li_sa_ri_files_from_trainer(trainer, frac_sacrum_arr)
        predicted_frac_RightIliac_arr, class_prob_frac_RightIliac_arr = predict_li_sa_ri_files_from_trainer(trainer, frac_RightIliac_arr)
        sys.stdout.write('<----------Fracture Segmentation Model completed---------------------> \n')
        # print('<----------Fracture Segmentation Model completed--------------------->')

        min_valid_object_size = 500

        # We know that getImageFromArray gives the default spacing of 1, but we have already performed transformation of spacing as 1 so, there is no problem there. 

        mask_preprocessed_arr = separate_labels_for_non_connected_splitted_fragments(predicted_frac_LeftIliac_arr, for_which_classes=None, volume_per_voxel=float(np.prod(original_img.GetSpacing(), dtype=np.float64)), minimum_valid_object_size=min_valid_object_size)
        li_mask = get_preds(mask_preprocessed_arr, 10)

        mask_preprocessed_arr = separate_labels_for_non_connected_splitted_fragments(predicted_frac_sacrum_arr, for_which_classes=None, volume_per_voxel=float(np.prod(original_img.GetSpacing(), dtype=np.float64)), minimum_valid_object_size=min_valid_object_size)
        sa_mask = get_preds(mask_preprocessed_arr, 0)

        mask_preprocessed_arr = separate_labels_for_non_connected_splitted_fragments(predicted_frac_RightIliac_arr, for_which_classes=None, volume_per_voxel=float(np.prod(original_img.GetSpacing(), dtype=np.float64)), minimum_valid_object_size=min_valid_object_size)
        ri_mask = get_preds(mask_preprocessed_arr, 20)

        overall_mask = return_one_with_max_probability(li_mask, sa_mask, ri_mask, li_prob = class_prob_frac_LeftIliac_arr, sa_prob = class_prob_frac_sacrum_arr, ri_prob=class_prob_frac_RightIliac_arr)

        overall_mask = overall_mask.astype(np.int8)
        print(np.unique(overall_mask, return_counts=True))
        print(overall_mask.shape)

        overall_mask_img = sitk.GetImageFromArray(overall_mask)
        overall_mask_img.SetSpacing(original_img.GetSpacing())
        overall_mask_img.SetDirection(original_img.GetDirection())
        overall_mask_img.SetOrigin(original_img.GetOrigin())

        print('<----------Upto Model orientation completed---------------------> \n')


        Path(OUTPUT_PATH / "images/pelvic-fracture-ct-segmentation").mkdir(parents=True, exist_ok=True)

        sitk.WriteImage(overall_mask_img, Path(OUTPUT_PATH / "images/pelvic-fracture-ct-segmentation") / f"{img_index}.mha", useCompression=True)

        spacing, gt_volume = load_gt_label_and_spacing(Path(train_label))

        metrics_single_case = evaluate_3d_single_case(gt_volume, overall_mask, spacing, verbose=True)

        print(metrics_single_case)

    return 0

if __name__ == "__main__":
    # import gdown 
    # gdown.download("https://drive.google.com/uc?id=1TDlfk8tGhMRIvk86nG8yspna2ZZ1-Lf0", join(RESOURCE_PATH, 'model_best.model'))
    raise SystemExit(run())


# PYTHONPATH=/path/to/Project python script.py
# python3 inference.py

