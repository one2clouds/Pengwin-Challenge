import pandas as pd
import glob 
from tqdm import tqdm 
import SimpleITK as sitk
import os 
import numpy as np

def check_voxels_resolution_and_size():
    img_list = sorted(glob.glob('/mnt/Enterprise2/shirshak/PENGWIN_TASK/PENGWIN_CT/part*/*.mha'))
    label_list = sorted(glob.glob('/mnt/Enterprise2/shirshak/PENGWIN_TASK/PENGWIN_CT/labels/*.mha'))

    # print(len(img_list))
    # print(len(label_list))

    columns = ['Name','image_shape_voxel', 'label_shape_voxel', 'unique_labels']
    df = pd.DataFrame(columns=columns)

    for index, image_and_mask in enumerate(tqdm(zip(img_list, label_list))):
        temp_image_sitk = sitk.ReadImage(image_and_mask[0])
        temp_image_arr = sitk.GetArrayFromImage(temp_image_sitk)
        voxel_size_image = temp_image_sitk.GetSpacing()
        
        temp_mask_sitk=sitk.ReadImage(image_and_mask[1])
        temp_mask_arr = sitk.GetArrayFromImage(temp_mask_sitk)
        voxel_size_mask = temp_mask_sitk.GetSpacing()
        # print(temp_image_arr.shape)
        # print(voxel_size_image)
        # print(temp_mask_arr.shape)
        # print(voxel_size_mask)
        df.loc[index+1, 'Name'] = os.path.split(image_and_mask[1])[1].split('.mha')[0]
        df.loc[index+1, 'image_shape_voxel'] = temp_image_arr.shape,voxel_size_image
        df.loc[index+1, 'label_shape_voxel'] =  temp_mask_arr.shape , voxel_size_mask
        df.loc[index+1, 'unique_labels'] = np.unique(temp_mask_arr, return_counts=True)

    return df

if __name__ == '__main__':
    df = check_voxels_resolution_and_size()
    print(df)
    df.to_csv('zzz_tests/sizes_and_voxels.csv')