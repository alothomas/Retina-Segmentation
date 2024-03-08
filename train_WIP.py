# %%
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


# %%

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice


# %%

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
            A.ToFloat(max_value=255.0),
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
        _, mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)


        image = np.stack([image] * 3, axis=-1)

        if self.augment:
            augmented = self.transform(image=image, mask=mask)
        else:
            # Apply ToTensorV2 and conversion to float for non-augmented path
            augmented = A.Compose([
            A.ToFloat(max_value=255.0),
            ToTensorV2(),
            ])(image=image, mask=mask)

        image = augmented['image']
        mask = augmented['mask']

        return image, mask.float()



# %%

# Data Loaders
BATCH_SIZE = 8
train_dataset = CustomDataset('data_model/train/images' , 'data_model/train/masks', augment=True)
val_dataset = CustomDataset('data_model/test/images', 'data_model/test/masks')

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Model, Optimizer, and Loss Functions
model = UNetWithResnet50Encoder(n_classes=1).to(device)

#model.load_state_dict(torch.load('output_dir2/model_epoch_1.pth'))
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)

#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, verbose=True)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
loss_fn = DiceLoss().to(device)



# %%
import matplotlib.pyplot as plt

def show_images_and_masks(dataset, num_imgs=3):
    fig, axs = plt.subplots(num_imgs, 2, figsize=(10, num_imgs * 5))
    
    for i in range(num_imgs):
        idx = np.random.randint(0, len(dataset))  # Randomly select an index
        image, mask = dataset[idx]
        
        # The image tensor might be normalized. Since we're skipping denormalization,
        # the image could appear with altered contrast/brightness.
        image = image.permute(1, 2, 0).numpy()  # Change from CxHxW to HxWxC for plotting
        
        mask = mask.squeeze().numpy()  # Remove channel dim for mask (C=1)
        
        # Plot image
        axs[i, 0].imshow(image, cmap='gray')
        axs[i, 0].set_title(f"Image {idx}")
        axs[i, 0].axis('off')
        
        # Plot mask
        axs[i, 1].imshow(mask, cmap='gray')
        axs[i, 1].set_title(f"Mask {idx}")
        axs[i, 1].axis('off')
    
    plt.show()



# Display images and masks
show_images_and_masks(train_dataset, num_imgs=2)


# %%

# Training Function
def train_one_epoch(epoch):
    model.train()
    loss_total = 0.0
    loss_throughout_epoch = []

    for input_img, mask in tqdm(train_loader, desc=f'Training epoch {epoch}'):
        input_img, mask = input_img.to(device), mask.to(device)

        optimizer.zero_grad()
        output = model(input_img)
        loss = loss_fn(output, mask)

        # Combine losses and backpropagate, normalize them against each other
        loss.backward()
        optimizer.step()

        # Accumulate individual losses for logging
        loss_total += loss.item()
        loss_throughout_epoch.append(loss.item())
    
    # Save graph of loss throughout epoch
    plt.figure(figsize=(12, 8))
    plt.plot(loss_throughout_epoch, label='Loss')
    plt.title('Loss')
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

            output = model(input_img)
            loss = loss_fn(output, mask)
            loss_total += loss.item()

    return loss_total / len(val_loader)


# Plot Loss Function
def plot_loss(epoch,train_inp_losses, val_inp_losses):
    plt.figure(figsize=(12, 8))

    # Ensure the x-axis matches the number of epochs
    epochs = range(1, epoch + 2)  # +1 for zero-based indexing, +1 for inclusive range

    plt.plot(epochs, train_inp_losses, label='Train Loss')
    plt.plot(epochs, val_inp_losses, label='Val Loss')
    plt.title('Loss')
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

    print(f'Epoch {epoch}, Train Loss: {train_inp_loss}')
    print(f'Epoch {epoch}, Val  Loss: {val_inp_loss}')

    save_model(model,f'st_output_dir/model_epoch_{epoch}.pth')
    plot_loss(epoch, train_inp_losses, val_inp_losses)

    

# %%
    

def visualize(image, true_mask=None, predicted_mask=None):
    """Visualize comparison between input image, true mask, and predicted mask."""
    fig, axs = plt.subplots(1, 3, figsize=(20, 10))  # Adjust the size as needed

    axs[0].imshow(image, cmap='gray')
    axs[0].set_title('Input Image')
    axs[0].axis('off')

    if true_mask is not None:
        axs[1].imshow(true_mask, cmap='gray')
        axs[1].set_title('True Mask')
        axs[1].axis('off')
    else:
        axs[1].axis('off')  # Hide the axis if true mask is not provided

    axs[2].imshow(predicted_mask, cmap='gray')
    axs[2].set_title('Predicted Mask')
    axs[2].axis('off')

    plt.show()


# %%

def predict_and_visualize(model, dataset, device):
    model.eval()  
    with torch.no_grad(): 
        for i in range(len(dataset)):
            input_img, true_mask = dataset[i]  
            input_img_unsqueeze = input_img.unsqueeze(0).to(device)  
            
            # Predict
            pred_mask = model(input_img_unsqueeze)
            pred_mask = pred_mask.squeeze().cpu().numpy().astype(np.uint8)
            pred_mask = pred_mask > 0.5  

            # Convert the input image and true mask to numpy for visualization, if available
            input_img_np = input_img.squeeze().permute(1, 2, 0).cpu().numpy()  # Assuming input_img is CxHxW
            input_img_np = (input_img_np * 0.5) + 0.5  # Assuming normalization was done with mean=0.5, std=0.5
            true_mask_np = true_mask.squeeze().cpu().numpy() if true_mask is not None else None

            # Use the visualize function
            visualize(
                image=input_img_np,
                true_mask=true_mask_np,
                predicted_mask=pred_mask
            )



# %%
real_data_dataset = CustomDataset('data_model/train/images', 'data_model/train/masks', augment=False)

# Load the model (ensure the model is already trained and weights are loaded)
model = UNetWithResnet50Encoder(n_classes=1).to(device)
model.load_state_dict(torch.load('st_output_dir\model_epoch_9.pth'))
# Predict and visualize on the real data
predict_and_visualize(model, real_data_dataset, device)
