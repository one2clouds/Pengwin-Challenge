import pandas as pd
import glob 
from tqdm import tqdm 
import SimpleITK as sitk
import os 
import numpy as np
import math


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

def check_voxels_resolution_and_size():
    label_list = sorted(glob.glob('/mnt/Enterprise2/shirshak/PENGWIN_TASK/PENGWIN_CT/labels/*.mha'))

    columns = ['Label','All Pixel Counts and Occurances']
    df = pd.DataFrame(columns=columns)

    pixel_counts_dict = dict()

    for mask in tqdm(label_list):
        temp_mask_sitk=sitk.ReadImage(mask)

        # print(temp_mask_sitk.GetSize())

        resampled_image = resample_img(temp_mask_sitk, (1.0,1.0,1.0), is_label=True)

        # print(resampled_image.GetSize())

        masks, pixels = np.unique(sitk.GetArrayFromImage(resampled_image), return_counts=True)

        # As 1 mm = 0.1 cm, h*w*d lai /10 3 times so, divided by 1000
        pixels_in_cm = pixels / 1000 

        # assert len(masks) == len(resampled_image.GetSize())

        # mask_value_list, pixel_value_list = list(), list()

        for mask_value, pixel_value in zip(masks, pixels_in_cm):
            if mask_value in pixel_counts_dict.keys():
                pixel_counts_dict[mask_value].append(pixel_value)
            else:
                pixel_counts_dict[mask_value] = []
                pixel_counts_dict[mask_value].append(pixel_value)


    # # Sort my dictionary
    pixel_counts_dict = dict(sorted(pixel_counts_dict.items()))

    for index, (key, value) in enumerate(pixel_counts_dict.items()):
        df.loc[index+1, 'Label'] = key
        df.loc[index+1, 'All Pixel Counts and Occurances'] = value
    return df

if __name__ == '__main__':
    df = check_voxels_resolution_and_size()
    print(df)
    df.to_csv('zzz_tests/get_all_size_as_list_new.csv')