import nibabel as nib
import numpy as np

file_path = 'project/ds000105_R2.0.2_raw/sub-1/func/sub-1_task-objectviewing_run-01_bold.nii.gz'

# Load the NIfTI file
image = nib.load(file_path)

