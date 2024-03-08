import matplotlib.pyplot as plt
from resnet50_unet import UNetWithResnet50Encoder
from train_WIP import CustomDataset
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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