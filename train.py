import os
import cv2
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2

from resnet50_unet import UNetWithResnet50Encoder

torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        #y_pred = torch.sigmoid(y_pred)   #apply sigmoid to clamp between 0 and 1
        
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)

        intersection = (y_pred * y_true).sum()
        dice = (2. * intersection + self.smooth) / (y_pred.sum() + y_true.sum() + self.smooth)

        return (1 - dice)

import os
import cv2
import numpy as np
from torch.utils.data import Dataset
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

class CustomDataset(Dataset):
    def __init__(self, image_dir, mask_dir, augment=True):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        # List image files, assuming they're named 'image_*.png'
        self.image_files = [f for f in os.listdir(image_dir) if f.startswith('image_')]
        self.augment = augment
        self.transform = A.Compose([
            A.OneOf([
                A.Rotate(limit=(180, 180), p=0.5),
                A.Rotate(limit=(90, 90), p=0.5),
            ], p=1.0),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
            A.Normalize(mean=(0.5,), std=(0.5,)),
            ToTensorV2(),
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Use the image filename to construct the corresponding mask filename
        image_filename = self.image_files[idx]
        # Extract ID from the image filename
        image_id = image_filename.split('_')[1].split('.')[0]
        mask_filename = f"mask_{image_id}.png"

        image_path = os.path.join(self.image_dir, image_filename)
        mask_path = os.path.join(self.mask_dir, mask_filename)
        
        # Load and preprocess the image and mask
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (512, 512))
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (512, 512))

        image = np.stack([image] * 3, axis=-1)

        if self.augment:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        else:
            image = np.expand_dims(image, axis=-1)  # Add channel dimension for albumentations
            augmented = A.Normalize(mean=(0.5,), std=(0.5,))(image=image)['image']
            image = ToTensorV2()(image=augmented)['image']
            mask = np.expand_dims(mask, axis=-1)  
            mask = ToTensorV2()(image=mask)['image']

        return image, mask.float()  


# helper function for data visualization
def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()

# Data Loaders
train_dataset = CustomDataset('data_model/train/images', 'data_model/train/masks', augment=True)
val_dataset = CustomDataset('data_model/test/images', 'data_model/test/masks')
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)



# Model, Optimizer, and Loss Functions
model = UNetWithResnet50Encoder(n_classes=1).to(device)
#model.load_state_dict(torch.load('output_dir2/model_epoch_1.pth'))
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, verbose=True)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
# dice loss
loss_fn = DiceLoss()


# Training Function
def train_one_epoch(epoch):
    model.train()
    loss_total = 0.0
    loss_throughout_epoch = []

    for input_img, mask in tqdm(train_loader, desc=f'Training epoch {epoch}'):
        input_img, mask = input_img.to(device), mask.to(device)

        optimizer.zero_grad()
        output = model.forward(input_img)

        loss = loss_fn(output, mask)

        # Combine losses and backpropagate, normalize them against each other
        loss.backward()
        optimizer.step()

        # Accumulate individual losses for logging
        loss_total += loss.item()
        loss_throughout_epoch.append(loss.item())
    
    # Save graph of loss throughout epoch
    plt.figure(figsize=(12, 8))
    plt.plot(loss_throughout_epoch, label='Inpainting Loss')
    plt.title('Inpainting Loss')
    plt.legend()
    plt.tight_layout()
    # Create the directory if it doesn't exist
    if not os.path.exists('st_output_dir'):
        os.makedirs('st_output_dir')

    plt.savefig(f'st_output_dir/internalloss_graph_epoch_{epoch}.png')
    plt.close()

    return loss_total / len(train_loader)

# Validation Function
def validate(epoch):
    model.eval()
    loss_total = 0.0

    with torch.no_grad():
        for input_img, mask in tqdm(val_loader, desc=f'Validating epoch {epoch}'):
            input_img, mask = input_img.to(device), mask.to(device)

            inpainting_output = model(input_img)
            loss = loss_fn(inpainting_output, mask)
            loss_total += loss.item()

    return loss_total / len(val_loader)


# Plot Loss Function
def plot_loss(epoch,train_inp_losses, val_inp_losses):
    plt.figure(figsize=(12, 8))

    # Ensure the x-axis matches the number of epochs
    epochs = range(1, epoch + 2)  # +1 for zero-based indexing, +1 for inclusive range

    plt.plot(epochs, train_inp_losses, label='Train Inpainting Loss')
    plt.plot(epochs, val_inp_losses, label='Val Inpainting Loss')
    plt.title('Inpainting Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'st_output_dir/loss_graph.png')
    plt.close()


# Save Model Function
def save_model(model, path):
    torch.save(model.state_dict(), path)


train_inp_losses = []
val_inp_losses = []

# Main Training Loop
num_epochs = 10
for epoch in range(num_epochs):
    train_inp_loss = train_one_epoch(epoch)
    scheduler.step(train_inp_loss)
    val_inp_loss = validate(epoch)
    train_inp_losses.append(train_inp_loss)
    val_inp_losses.append(val_inp_loss)

    print(f'Epoch {epoch}, Train Inpainting Loss: {train_inp_loss}')
    print(f'Epoch {epoch}, Val Inpainting Loss: {val_inp_loss}')

    save_model(model,f'st_output_dir/model_epoch_{epoch}.pth')
    plot_loss(epoch, train_inp_losses, val_inp_losses)

    