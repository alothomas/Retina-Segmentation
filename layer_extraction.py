from skimage import exposure
import numpy as np
import czifile, tifffile
import os
from skimage import feature, color, io
import matplotlib.pyplot as plt

def process_images(image_dir, output_dir):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # List all CZI files in the directory
    image_paths = [os.path.join(image_dir, filename) for filename in os.listdir(image_dir) if filename.endswith('.czi')]

    # Define the function to calculate LBP entropy
    def calculate_lbp(image, points=24, radius=8):
        lbp = feature.local_binary_pattern(image, points, radius, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, points + 3), range=(0, points + 2))
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-7)
        return -1 * (hist * np.log2(hist + 1e-7)).sum()

    for image_file in image_paths:
        lbp_entropies = []

        try:
            print(f"Computing LBP for {image_file}")
            with czifile.CziFile(image_file) as czi:
                image_array = czi.asarray()

            for stack_index in range(image_array.shape[4]):
                stack = image_array[0, 0, 0, 0, stack_index, :, :, 0]
                contrast_img = exposure.rescale_intensity(stack, in_range='image')
                contrast_img = exposure.equalize_hist(contrast_img)
                lbp_entropy = calculate_lbp(contrast_img)
                lbp_entropies.append((stack_index, lbp_entropy, contrast_img))
                
        except Exception as e:
            print(f"Error processing file {image_file}: {e}")

        lbp_entropies.sort(key=lambda x: x[1], reverse=True)

        try:
            top_stack_index, _, top_stack = lbp_entropies[0]
            filename = os.path.basename(image_file).replace(".czi", f"_stack_{top_stack_index}.tif")
            tifffile.imsave(os.path.join(output_dir, filename), top_stack)
            print(f"Saved {filename} to {output_dir}")

        except Exception as e:
            print(f"Error saving top stack: {e}")

# Define the directories that contain your images
image_dirs = [r'E:\Akita\1 Traditional Area Fraction Vessel Density\FG12_NG004_randomizer\Original Stack czi',
              r'E:\Akita\1 Traditional Area Fraction Vessel Density\PBS_11C7_randamizer\Original Stack Czi']
output_dirs = [r'data/FG12', r'data/PBS']

# Process images for each directory
for image_dir, output_dir in zip(image_dirs, output_dirs):
    process_images(image_dir, output_dir)

print("Done!")
