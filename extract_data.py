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
    pic_num = 9  # number of pictures for each object each run
    offset = 1 # shift for response delay
    z_value = [27, 29, 31, 33, 35]
    x_train = np.zeros((40, 64, 5, 198)) # (x, y, z, number of pictures)  198 = 2*9*11
    y_train = np.zeros(198)
    x_test = np.zeros((40, 64, 5, 2*pic_num))
    y_test = np.zeros(2*pic_num)
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
        
        # Get object start number
        obj0s, obj0e, obj1s, obj1e = filter[x-1, :]
        obj0s += offset
        obj1s += offset
        i = 0

        for z in z_value:
            x_train[:, :, i, 2*pic_num*(x-1) : 2*pic_num*(x-1)+pic_num] = data[:, :, z, obj0s : obj0s+pic_num]
            y_train[2*pic_num*(x-1) : 2*pic_num*(x-1)+pic_num] = 0
            x_train[:, :, i, 2*pic_num*(x-1)+pic_num : 2*pic_num*(x-1)+2*pic_num] = data[:, :, z, obj1s : obj1s+pic_num]
            y_train[2*pic_num*(x-1)+pic_num : 2*pic_num*(x-1)+2*pic_num] = 1
            i += 1

    # Create testing data
    file_path = fp + '/sub-1/func/sub-1_task-objectviewing_run-12_bold.nii.gz'

    image = nib.load(file_path)
    data = image.get_fdata()

    obj0s, obj0e, obj1s, obj1e = filter[11, :]
    obj0s += offset
    obj1s += offset
    i = 0
    
    for z in z_value:
        x_test[:, :, i, 0:pic_num] = data[:, :, z, obj0s:obj0s+pic_num]
        y_test[0:pic_num] = 0
        x_test[:, :, i, pic_num:2*pic_num] = data[:, :, z, obj1s:obj1s+pic_num]
        y_test[pic_num:2*pic_num] = 1
        i += 1

    return x_train, y_train, x_test, y_test


# x_train, y_train, x_test, y_test = extract_data('')
# print(y_train)
# print(np.count_nonzero(x_train == 0))