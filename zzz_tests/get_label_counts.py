import pandas as pd
import glob 
from tqdm import tqdm 
import SimpleITK as sitk
import os 
import numpy as np

def check_voxels_resolution_and_size():
    img_list = sorted(glob.glob('/mnt/Enterprise2/shirshak/PENGWIN_TASK/PENGWIN_CT/part*/*.mha'))
    label_list = sorted(glob.glob('/mnt/Enterprise2/shirshak/PENGWIN_TASK/PENGWIN_CT/labels/*.mha'))

    columns = ['Label','Occurances']
    df = pd.DataFrame(columns=columns)

    dict_values = dict()

    for mask in tqdm(label_list):
        temp_mask_sitk=sitk.ReadImage(mask)
        temp_mask_arr = sitk.GetArrayFromImage(temp_mask_sitk)

        for mask_values in np.unique(temp_mask_arr):
            if mask_values in dict_values.keys():
                dict_values[mask_values] += 1
            else: 
                dict_values[mask_values] = 1

    # Sort my dictionary
    dict_values = dict(sorted(dict_values.items()))

    for index, (key, value) in enumerate(dict_values.items()):
        df.loc[index+1, 'Label'] = key
        df.loc[index+1, 'Occurances'] = value
    return df

if __name__ == '__main__':
    df = check_voxels_resolution_and_size()
    print(df)
    # df.to_csv('zzz_tests/occurances_of_labels.csv')