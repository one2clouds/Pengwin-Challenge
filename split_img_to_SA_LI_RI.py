import os 
import SimpleITK as sitk 
import glob 
from tqdm import tqdm 
import numpy as np


def maybe_mkdir_p(directory: str) -> None:
    os.makedirs(directory, exist_ok=True)


# input: ct_origin_fileName,ct_label_fileName,label output:frac_Grayscale==label
def extractSingleFrac(ct_origin_arr, ct_label_arr, label):
    frac_Grayscale_arr = ct_origin_arr.clone() # .copy() for array .clone() for tensor

    if label == 1:
        frac_Grayscale_arr[ct_label_arr != 1] = 0

    if label == 2:
        frac_Grayscale_arr[ct_label_arr != 2] = 0

    if label == 3:
        frac_Grayscale_arr[ct_label_arr != 3] = 0

    return frac_Grayscale_arr



def saveDiffFrac(ct_origin_arr, ct_label_arr):
    # label = 1: Sacrum / 2: Left Hip / 3:Right Hip
    frac_sacrum_arr = extractSingleFrac(ct_origin_arr, ct_label_arr, 1)
    frac_LeftIliac_arr = extractSingleFrac(ct_origin_arr, ct_label_arr, 2)
    frac_RightIliac_arr = extractSingleFrac(ct_origin_arr, ct_label_arr, 3)

    return frac_LeftIliac_arr, frac_sacrum_arr, frac_RightIliac_arr


