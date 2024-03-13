import numpy as np
from PIL import Image
import random
import matplotlib.pyplot as plt
import os
from tifffile import tifffile



def extract_overlapping_patches(image, patch_size=600, overlap=0.3):
    """
    Extracts overlapping patches of a given size from an image.
    
    :param image: Numpy array of the image.
    :param patch_size: The size of the square patches.
    :param overlap: Fraction of the patch size to overlap (e.g., 0.5 for 50% overlap).
    :return: List of image patches.
    """
    stride = int(patch_size * (1 - overlap))
    patches = []
    
    for y in range(0, image.shape[0] - patch_size + 1, stride):
        for x in range(0, image.shape[1] - patch_size + 1, stride):
            patch = image[y:y+patch_size, x:x+patch_size]
            patches.append(patch)
            
    return patches


def normalize_patch(patch):
    """Normalize image patch to the range [0, 1]."""
    patch_min = patch.min()
    patch_max = patch.max()
    if patch_max == patch_min:

        normalized_patch = np.zeros_like(patch)
    else:
        normalized_patch = (patch - patch_min) / (patch_max - patch_min)
    return normalized_patch


# Define your directories
input_dirs = ['data/FG12', 'data/PBS']
output_base = 'data/data_patch'

# Define the patch extraction parameters
patch_size = 512
overlap = 0.3

for input_dir in input_dirs:
    filenames = [f for f in os.listdir(input_dir) if f.endswith('.tif')]
    
    dir_name = os.path.basename(input_dir)
    output_dir_for_current_input = os.path.join(output_base, dir_name)
    if not os.path.exists(output_dir_for_current_input):
        os.makedirs(output_dir_for_current_input)

    for filename in filenames:
        file_path = os.path.join(input_dir, filename)
        image = tifffile.imread(file_path)
        patches = extract_overlapping_patches(image, patch_size, overlap)
        normalized_patches = [normalize_patch(patch) for patch in patches]

        base_filename = os.path.splitext(os.path.basename(filename))[0]
        output_dir_for_file = os.path.join(output_dir_for_current_input, base_filename)
        if not os.path.exists(output_dir_for_file):
            os.makedirs(output_dir_for_file)
        
        for i, patch in enumerate(normalized_patches):
            patch_image = Image.fromarray((patch * 255).astype(np.uint8))
            patch_image.save(os.path.join(output_dir_for_file, f'{base_filename}_patch_{i}.png'), 'PNG')
