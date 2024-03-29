import os
import cv2
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class PredCustomDataset(Dataset):
    def __init__(self, image_paths=None, image_dir=None, mask_dir=None, augment=False, denoise=False, inference=False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.denoise = denoise
        self.inference = inference
        self.augment = augment if not inference else False  
        
        if inference and image_paths is not None:
            self.image_files = image_paths 
        else:
            self.image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.png')]

        # Basic transformation pipeline
        self.basic_transform = A.Compose([
            A.Resize(512, 512),
            A.ToFloat(max_value=255.0),
            ToTensorV2(),
        ])

        # Augmentation transformation pipeline
        if not inference:
            self.augment_transform = A.Compose([
                A.OneOf([
                    A.Rotate(limit=180, p=0.5),
                    A.ElasticTransform(alpha=2, sigma=50, alpha_affine=50, p=0.5),
                ], p=1.0),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.GaussianBlur(blur_limit=3, p=0.5),
                A.GaussNoise(var_limit=(10, 50), p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomResizedCrop(height=512, width=512, scale=(0.8, 1.0), ratio=(0.75, 1.33), p=0.5),
                A.ToFloat(max_value=255.0),
                ToTensorV2(),
            ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if self.denoise:
            image = cv2.fastNlMeansDenoising(image, None, h=15, templateWindowSize=7, searchWindowSize=21)

        image = np.stack([image] * 3, axis=-1)

        if self.inference:
            transformed = self.basic_transform(image=image)
            image = transformed['image']
        else:
            mask_path = self.mask_dir + '/' + os.path.basename(image_path).replace('image', 'mask')
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)[1]

            if self.augment:
                transformed = self.augment_transform(image=image, mask=mask)
            else:
                transformed = self.basic_transform(image=image, mask=mask)
                
            image, mask = transformed['image'], transformed['mask']
            return image, mask.float()

        return image
    




def reconstruct_image_from_patches(predicted_patches, image_shape, patch_size=600, overlap=0.3):
    """
    Reconstructs an image from a set of predicted patches.

    Args:
        predicted_patches (list): List of predicted patches.
        image_shape (tuple): Shape of the original image.
        patch_size (int, optional): Size of each patch. Defaults to 600.
        overlap (float, optional): Overlap between patches as a fraction of patch size. Defaults to 0.3.

    Returns:
        numpy.ndarray: Reconstructed image as a numpy array.
    """
    stride = int(patch_size * (1 - overlap))
    reconstructed_image = np.zeros(image_shape, dtype=np.float32)
    
    idx = 0
    for y in range(0, image_shape[0] - patch_size + 1, stride):
        for x in range(0, image_shape[1] - patch_size + 1, stride):
            patch = predicted_patches[idx]
            # Use a logical OR operation for the overlap handling in binary masks
            reconstructed_image[y:y+patch_size, x:x+patch_size] = np.logical_or(reconstructed_image[y:y+patch_size, x:x+patch_size], patch)
            idx += 1

    return reconstructed_image.astype(np.float32)




