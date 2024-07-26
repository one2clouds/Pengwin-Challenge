import pandas as pd
import glob 
from tqdm import tqdm 
import SimpleITK as sitk
import os 
import numpy as np

def check_voxels_resolution_and_size():
    label_list = sorted(glob.glob('/mnt/Enterprise2/shirshak/PENGWIN_TASK/PENGWIN_CT/labels/*.mha'))

    columns = ['Label','Mean Pixel Counts and Occurances']
    df = pd.DataFrame(columns=columns)

    pixel_counts_dict = dict()

    for mask in tqdm(label_list):
        temp_mask_sitk=sitk.ReadImage(mask)
        temp_mask_arr = sitk.GetArrayFromImage(temp_mask_sitk)

        masks, pixels = np.unique(temp_mask_arr, return_counts=True)

        assert len(masks) == len(pixels)

        # mask_value_list, pixel_value_list = list(), list()

        for mask_value, pixel_value in zip(masks, pixels):
            if mask_value in pixel_counts_dict.keys():
                pixel_counts_dict[mask_value].append(pixel_value)
            else:
                pixel_counts_dict[mask_value] = []
                pixel_counts_dict[mask_value].append(pixel_value)


    # # Sort my dictionary
    pixel_counts_dict = dict(sorted(pixel_counts_dict.items()))

    for index, (key, value) in enumerate(pixel_counts_dict.items()):
        df.loc[index+1, 'Label'] = key
        df.loc[index+1, 'Mean Pixel Counts and Occurances'] = value
    return df

if __name__ == '__main__':
    df = check_voxels_resolution_and_size()
    print(df)
    df.to_csv('zzz_tests/get_all_size_as_list.csv')