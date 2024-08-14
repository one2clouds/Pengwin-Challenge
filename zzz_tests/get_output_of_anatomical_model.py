import SimpleITK as sitk 
import numpy as np 






if __name__ == '__main__':
    li_mask= sitk.ReadImage('/mnt/Enterprise2/shirshak/PENGWIN_TASK/PENGWIN_CT/nifti_fragments_LI_SA_RI/labels/001_LI_label.nii.gz')
    sa_mask = sitk.ReadImage('/mnt/Enterprise2/shirshak/PENGWIN_TASK/PENGWIN_CT/nifti_fragments_LI_SA_RI/labels/001_SA_label.nii.gz')
    ri_mask = sitk.ReadImage('/mnt/Enterprise2/shirshak/PENGWIN_TASK/PENGWIN_CT/nifti_fragments_LI_SA_RI/labels/001_RI_label.nii.gz')

    overall_mask = li_mask + sa_mask + ri_mask

    sitk.WriteImage(overall_mask, '/home/shirshak/Pengwin_Submission_Portal/zzz_tests/overall_mask.mha') 