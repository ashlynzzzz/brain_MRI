import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib

def display_mri_slices(file_path, slice_indices):
    """
    Display multiple slices from an MRI image in NIfTI format.

    Parameters:
    file_path (str): The file path to the NIfTI file.
    slice_indices (list of int): The indices of the slices to display.
    """
    # Load the NIfTI file
    image = nib.load(file_path)
    # Get the data array from the image
    data = image.get_fdata()
    
    # Determine the layout of the subplots
    subplot_dim = int(np.ceil(np.sqrt(len(slice_indices))))
    
    # Create a figure with subplots
    fig, axes = plt.subplots(subplot_dim, subplot_dim, figsize=(15, 15))
    
    # Flatten the axes array for easy iteration
    axes_flat = axes.flatten()
    
    # Loop through the list of slice indices
    for i, slice_index in enumerate(slice_indices):
        # Extract the specific slice to display
        slice_data = data[:, :, slice_index, 0]
        
        # Display the slice
        axes_flat[i].imshow(slice_data.T, cmap="gray", origin="lower")
        axes_flat[i].set_title(f'Slice {slice_index}')
        axes_flat[i].axis('off')  # Turn off axis
    
    # Turn off any unused subplots
    for j in range(i+1, len(axes_flat)):
        axes_flat[j].axis('off')
    
    # Show the full figure with all subplots
    plt.tight_layout()
    plt.show()

# List of slice indices that you want to display
slice_indices = list(range(27, 35))

# Call the function with the path to your NIfTI file and the list of slice indices
display_mri_slices('project/ds000105_R2.0.2_raw/sub-1/func/sub-1_task-objectviewing_run-08_bold.nii.gz', slice_indices)

