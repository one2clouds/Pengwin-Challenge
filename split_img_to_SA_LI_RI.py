import os 
import SimpleITK as sitk 
import glob 
from tqdm import tqdm 
import numpy as np


def maybe_mkdir_p(directory: str) -> None:
    os.makedirs(directory, exist_ok=True)


# input: ct_origin_fileName,ct_label_fileName,label output:frac_Grayscale==label
def extractSingleFrac(ct_scale_img, ct_label_img, label):
    # ct_origin_img = sitk.ReadImage(ct_origin_fileName)
    # ct_label_img = sitk.ReadImage(ct_label_fileName)
    ct_origin_arr = sitk.GetArrayFromImage(ct_scale_img)  # get array from image
    ct_label_arr = sitk.GetArrayFromImage(ct_label_img)  # get array from image
    # ct_label_arr = np.swapaxes(ct_label_arr, 0, 2)

    # print(ct_origin_arr.shape) # (286, 238, 459)
    # print(ct_scale_img.GetSize()) # (459, 238, 286)
    frac_Grayscale_img = ct_origin_arr.copy()

    if label == 1:
        frac_Grayscale_img[ct_label_arr != 1] = 0

    if label == 2:
        frac_Grayscale_img[ct_label_arr != 2] = 0

    if label == 3:
        frac_Grayscale_img[ct_label_arr != 3] = 0

    frac_Grayscale_img = sitk.GetImageFromArray(frac_Grayscale_img)

    frac_Grayscale_img.SetDirection(ct_scale_img.GetDirection())
    frac_Grayscale_img.SetSpacing(ct_scale_img.GetSpacing())
    frac_Grayscale_img.SetOrigin(ct_scale_img.GetOrigin())

    return frac_Grayscale_img



def saveDiffFrac(ct_origin_img, ct_label_img):
    # label = 1: Sacrum / 2: Left Hip / 3:Right Hip
    frac_sacrum_img = extractSingleFrac(ct_origin_img, ct_label_img, 1)
    frac_LeftIliac_img = extractSingleFrac(ct_origin_img, ct_label_img, 2)
    frac_RightIliac_img = extractSingleFrac(ct_origin_img, ct_label_img, 3)

    return frac_LeftIliac_img, frac_sacrum_img, frac_RightIliac_img


