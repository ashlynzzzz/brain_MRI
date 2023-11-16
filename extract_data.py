import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np

def extract_data(fp):
    '''
    Input: file path to ds000105_R2.0.2_raw folder
    Output: x_train, y_train, x_test, y_test
    object0: scissor
    object1: shoe
    9 pictures for each run each object
    '''
    z_value = 31
    x_train = np.zeros((40, 64, 198)) # (x, y, number of pictures)  198 = 2*9*11
    y_train = np.zeros(198)
    x_test = np.zeros((40, 64, 18))
    y_test = np.zeros(18)
    # object location for 12 runs
    filter = np.array([[5, 0, 48, 0],   # 12, 120
                        [63, 0, 34, 0],  # 156, 84
                        [34, 0, 77, 0],   # 84, 192
                        [106, 0, 5, 0],   # 264, 12
                        [20, 0, 77, 0],   # 48, 192
                        [106, 0, 48, 0],   # 264, 120
                        [34, 0, 48, 0],   # 84, 120
                        [34, 0, 48, 0],   # 84, 120
                        [106, 0, 77, 0],   # 264, 192
                        [63, 0, 92, 0],   # 156, 228
                        [106, 0, 92, 0],   # 264, 156
                        [106, 0, 77, 0]])  # 264, 192 

    # Create training data
    for x in range(1, 12):

        if(x < 10):
            y = "0" + str(x)
        else:
            y = str(x)
        # File path
        file_path = fp + f'/sub-1/func/sub-1_task-objectviewing_run-{y}_bold.nii.gz'

        # Load the data
        image = nib.load(file_path)
        data = image.get_fdata()
        
        obj0s, obj0e, obj1s, obj1e = filter[x-1, :]
        
        x_train[:, :, 18*(x-1):18*(x-1)+9] = data[:, :, z_value, obj0s:obj0s+9]
        y_train[18*(x-1):18*(x-1)+9] = 0
        x_train[:, :, 18*(x-1)+9:18*(x-1)+18] = data[:, :, z_value, obj1s:obj1s+9]
        y_train[18*(x-1)+9:18*(x-1)+18] = 1
        
    # Create testing data
    file_path = fp + '/sub-1/func/sub-1_task-objectviewing_run-12_bold.nii.gz'

    image = nib.load(file_path)
    data = image.get_fdata()

    obj0s, obj0e, obj1s, obj1e = filter[11, :]

    x_test[:, :, 0:9] = data[:, :, z_value, obj0s:obj0s+9]
    y_test[0:9] = 0
    x_test[:, :, 9:18] = data[:, :, z_value, obj1s:obj1s+9]
    y_test[9:18] = 1

    return x_train, y_train, x_test, y_test


x_train, y_train, x_test, y_test = extract_data('project/ds000105_R2.0.2_raw')
print(y_train)
print(np.count_nonzero(x_train == 0))