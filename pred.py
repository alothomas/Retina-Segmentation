import os
import re
import numpy as np
import cv2
from PIL import Image
import torch
from torch.utils.data import DataLoader
from tifffile import imread, imsave
import tifffile

from resnet50_unet import UNetWithResnet50Encoder
from utils_pred import PredCustomDataset, reconstruct_image_from_patches


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()



def predict_image(loader, model, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for images in loader:
            images = images.to(device)
            output = model(images)
            output = torch.sigmoid(output)
            pred = output > 0.5
            pred = pred.squeeze(1)
            preds.extend(pred.detach().cpu().numpy())
    return preds


def process_folders(input_folders, output_base, model_path, patch_size=512, overlap=0.3, batch_size=8):
    for input_folder in input_folders:
        original_image_folder = os.path.join('data', os.path.basename(input_folder))
        print(f'Processing {input_folder}')
        for image_name in os.listdir(original_image_folder):
            if image_name.endswith('.tif'):
                base_name = os.path.splitext(image_name)[0]
                current_path = os.path.join(input_folder, base_name)
                image_files = [f for f in os.listdir(current_path) if f.endswith('.png')]
                
                # Sort files by patch number
                patch_files = [(f, int(re.search(r'patch_(\d+)', f).group(1))) for f in image_files]
                patch_files.sort(key=lambda x: x[1])
                patches = [os.path.join(current_path, f[0]) for f in patch_files]
                
                if not patches:
                    continue
                
                model = UNetWithResnet50Encoder(n_classes=1).to(device)
                model.load_state_dict(torch.load(model_path))
                
                dataset = PredCustomDataset(image_paths=patches, inference=True, denoise=True)
                loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
                preds = predict_image(loader, model, device)
                
                original_image_path = os.path.join(original_image_folder, image_name)
                original_image = tifffile.imread(original_image_path)
                
                # Use the shape of the original image for reconstruction
                full_image = reconstruct_image_from_patches(preds, original_image.shape, patch_size=patch_size, overlap=overlap)
                
                output_dir = os.path.join(output_base, os.path.basename(input_folder), base_name)
                os.makedirs(output_dir, exist_ok=True)
                
                with tifffile.TiffFile(original_image_path) as tif:
                    metadata = {tag.name: tag.value for tag in tif.pages[0].tags.values()}
                
                # Save the reconstructed image and other processed images
                imsave(os.path.join(output_dir, f'{base_name}_reconstructed.tif'), full_image, metadata=metadata)
                
                
                if original_image.dtype != np.uint8:
                    original_image_8bit = ((original_image - original_image.min()) / (original_image.max() - original_image.min()) * 255).astype(np.uint8)
                else:
                    original_image_8bit = original_image

                if full_image.dtype != np.uint8:
                    full_image_8bit = ((full_image - full_image.min()) / (full_image.max() - full_image.min()) * 255).astype(np.uint8)
                
                _, original_image_binary = cv2.threshold(original_image_8bit, 127, 255, cv2.THRESH_BINARY)
                _, full_image_binary = cv2.threshold(full_image_8bit, 127, 255, cv2.THRESH_BINARY)

                # Perform binary subtraction and ensure results stay within binary limits [0, 255] for visual representation
                leakage_areas = np.clip(original_image_binary - full_image_binary, 0, 255)
                unperfused_vessels = np.clip(full_image_binary - original_image_binary, 0, 255)
                perfused_vessels = np.clip(full_image_binary & (~unperfused_vessels), 0, 255)

                # Save the results
                imsave(os.path.join(output_dir, f'{base_name}_leakage_areas.tif'), leakage_areas, metadata=metadata)
                imsave(os.path.join(output_dir, f'{base_name}_unperfused_vessels.tif'), unperfused_vessels, metadata=metadata)
                imsave(os.path.join(output_dir, f'{base_name}_perfused_vessels.tif'), perfused_vessels, metadata=metadata)

                # Combined image of perfused (red) and unperfused (green) vessels
                combined_vessels = np.zeros((original_image.shape[0], original_image.shape[1], 3), dtype=np.uint8)
                combined_vessels[..., 0] = perfused_vessels  
                combined_vessels[..., 2] = unperfused_vessels

                imsave(os.path.join(output_dir, f'{base_name}_perfusion_combined.tif'), combined_vessels, metadata=metadata)

                original_image_rgb = cv2.cvtColor(original_image_8bit, cv2.COLOR_GRAY2RGB)                

                full_image_resized = cv2.resize(full_image, (original_image_rgb.shape[1], original_image_rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
                red_overlay = np.zeros_like(original_image_rgb)
                red_overlay[full_image_resized == 1] = [255, 0, 0]
                combined_image = cv2.addWeighted(red_overlay, 1, original_image_rgb, 0.7, 0)

                Image.fromarray(combined_image).save(os.path.join(output_dir, f'{base_name}_overlay.png'))



input_folders = ['data\\data_patch\\FG12', 'data\\data_patch\\PBS']
output_base = 'data\\data_processed'
model_path = 'resnetunet\\st_output_dir\\best_model_unet_40_epochs_UNET.pth'

process_folders(input_folders, output_base, model_path, patch_size=512, overlap=0.3, batch_size=8)
print('Done')
