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

    mask_counts_dict = dict()
    pixel_counts_dict = dict()

    for mask in tqdm(label_list[:5]):
        temp_mask_sitk=sitk.ReadImage(mask)
        temp_mask_arr = sitk.GetArrayFromImage(temp_mask_sitk)

        masks, pixels = np.unique(temp_mask_arr, return_counts=True)

        assert len(masks) == len(pixels)

        for mask_value, pixel_value in zip(masks, pixels):
            if mask_value in mask_counts_dict.keys():
                mask_counts_dict[mask_value] += 1
                pixel_counts_dict[mask_value] += pixel_value
            else: 
                mask_counts_dict[mask_value] = 1
                pixel_counts_dict[mask_value] = pixel_value

    # Dictionary Values Division
    mean_pixel_count_dict = {key: pixel_counts_dict[key] // mask_counts_dict.get(key, 0) for key in pixel_counts_dict.keys()}

    # # Sort my dictionary
    mean_pixel_count_dict = dict(sorted(mean_pixel_count_dict.items()))

    for index, (key, value) in enumerate(mean_pixel_count_dict.items()):
        df.loc[index+1, 'Label'] = key
        df.loc[index+1, 'Mean Pixel Counts and Occurances'] = value
    return df

if __name__ == '__main__':
    df = check_voxels_resolution_and_size()
    print(df)
    # df.to_csv('zzz_tests/get_overall_size.csv')