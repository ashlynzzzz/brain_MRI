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
    offset = 2 # shift for response delay
    z_value = [27, 29, 31, 33, 35]
    x_data = np.zeros((40, 64, 5, 2*2*9*12)) # (x, y, z, number of pictures)  432 = 2*2*9*12
    y_data = np.zeros(2*2*9*12)
    x_train = np.zeros((40, 64, 5, 2*2*9*11))
    y_train = np.zeros(2*2*9*11)
    x_test = np.zeros((40, 64, 5, 2*2*9*1))
    y_test = np.zeros(2*2*9*1)
    # object location for 12 runs, obj0: scissor, obj1: shoe
    filter_sci_sho = np.array([[5, 0, 48, 0],   # 12, 120
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
                        [106, 0, 77, 0], # 264, 192 
                        # sub2
                        [5, 0, 48, 0],   # 12, 120
                        [34, 0, 48, 0],  # 84, 120
                        [106, 0, 77, 0],   # 264, 192
                        [106, 0, 5, 0],   # 264, 12
                        [63, 0, 34, 0],   # 156, 84
                        [34, 0, 77, 0],   # 84, 192
                        [20, 0, 77, 0],   # 48, 192
                        [34, 0, 48, 0],   # 84, 120
                        [63, 0, 92, 0],   # 156, 228
                        [106, 0, 48, 0],   # 264, 120
                        [106, 0, 92, 0],   # 264, 156
                        [106, 0, 77, 0]])  # 264, 192 
    # object location for 12 runs, obj0: scissor, obj1: cat
    filter_sci_cat = np.array([[5, 0, 34, 0],   # 12, 84
                        [63, 0, 20, 0],  # 156, 48
                        [34, 0, 5, 0],   # 84, 12
                        [106, 0, 48, 0],   # 264, 120
                        [20, 0, 92, 0],   # 48, 228
                        [106, 0, 77, 0],   # 264, 192
                        [34, 0, 92, 0],   # 84, 228
                        [34, 0, 77, 0],   # 84, 192
                        [106, 0, 34, 0],   # 264, 84
                        [63, 0, 20, 0],   # 156, 48
                        [106, 0, 5, 0],   # 264, 12
                        [106, 0, 92, 0], # 264, 228 
                        # sub2
                        [5, 0, 34, 0],   # 12, 84
                        [34, 0, 77, 0],  # 84, 192
                        [106, 0, 34, 0],   # 264, 84
                        [106, 0, 48, 0],   # 264, 120
                        [63, 0, 20, 0],   # 156, 48
                        [34, 0, 5, 0],   # 84, 12
                        [20, 0, 92, 0],   # 48, 228
                        [34, 0, 92, 0],   # 84, 228
                        [63, 0, 20, 0],   # 156, 48
                        [106, 0, 77, 0],   # 264, 192
                        [106, 0, 5, 0],   # 264, 12
                        [106, 0, 92, 0]])  # 264, 228 
    

    # Create training data
    # Go through all the sub
    for s in range(1, 3):
        
        # Go through all the run
        for x in range(1, 13):

            if(x < 10):
                y = "0" + str(x)
            else:
                y = str(x)
            # File path
            file_path = fp + f'/sub-{s}/func/sub-{s}_task-objectviewing_run-{y}_bold.nii.gz'

            # Load the data
            image = nib.load(file_path)
            data = image.get_fdata()
            
            # Get object start number
            obj0s, obj0e, obj1s, obj1e = filter_sci_cat[12*(s-1) + x-1, :]
            obj0s += offset
            obj1s += offset
            i = 0

            # Go through all the slice
            for z in z_value:
                x_data[:, :, i, 216*(s-1) + 2*pic_num*(x-1) : 216*(s-1) + 2*pic_num*(x-1)+pic_num] = data[:, :, z, obj0s : obj0s+pic_num]
                y_data[216*(s-1) + 2*pic_num*(x-1) : 216*(s-1) + 2*pic_num*(x-1)+pic_num] = 0
                x_data[:, :, i, 216*(s-1) + 2*pic_num*(x-1)+pic_num : 216*(s-1) + 2*pic_num*(x-1)+2*pic_num] = data[:, :, z, obj1s : obj1s+pic_num]
                y_data[216*(s-1) + 2*pic_num*(x-1)+pic_num : 216*(s-1) + 2*pic_num*(x-1)+2*pic_num] = 1
                i += 1

    # Shuffle the data set
    # Generate a permutation of indices
    perm = np.random.permutation(2*2*9*12)
    # Apply the permutation to both arrays
    x_data = x_data[:, :, :, perm]
    y_data = y_data[perm]
    
    # take 11/12 as training set
    x_train = x_data[:, :, :, 0: 397]
    y_train = y_data[0: 397]
    # take 1/12 as testing set
    x_test = x_data[:, :, :, 397: 433]
    y_test = y_data[397: 433]
    
    return x_train, y_train, x_test, y_test


#x_train, y_train, x_test, y_test = extract_data('project/ds000105_R2.0.2_raw')
#print(y_train)
#print(np.count_nonzero(y_train))
#print(np.count_nonzero(y_test))