"""
U-Net Implementation in PyTorch

U-Nets are a type of convolutional neural network originally designed for biomedical image segmentation. 
They are characterized by their unique architecture, which resembles the letter "U". This architecture 
consists of two main parts: an encoder (contracting path) and a decoder (expansive path).

Encoder:
- The encoder is typically a sequence of convolutional layers followed by max-pooling layers. 
- It captures the context in the input image, reducing its spatial dimensions while increasing the depth.

Decoder:
- The decoder consists of upsampled layers and convolutional layers.
- It progressively recovers the spatial resolution, focusing on precise localization to provide accurate segmentation.

Skip Connections:
- A key feature of U-Nets are the skip connections between the encoder and decoder.
- These connections pass feature maps from the encoder to the decoder, combining low-level features with high-level ones.
- This helps in precise localization, a critical aspect in segmentation tasks.

In this implementation:
- The U-Net is built to work with general image input channels and output channels.
- The initial number of filters in the first layer of the encoder is set to 64, doubling with each downsampling step.
- In the bottleneck, the number of filters reaches 1024 before the upsampling process starts in the decoder.
- The final layer reduces the number of output channels, making it suitable for binary or multi-class segmentation.

This implementation is flexible and can be adapted for different segmentation tasks by adjusting the input and output channels.

Usage:
- The model can be used for training with pairs of images and their corresponding segmentation masks.
- During inference, it predicts segmentation masks for unseen images.

"""

# Importing necessary modules from PyTorch.
import os
import torch  # PyTorch library, provides tensor operations and other utilities.
import torch.optim as optim
import torch.nn as nn  # nn is a sub-module in PyTorch for building neural networks.
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
import sys, select
from random import shuffle

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def estimate_batch_size(model, input_shape, target_memory_usage):
    # Estimate the memory footprint of a single sample
    sample_input = torch.randn(input_shape).to(device)
    sample_output = model(sample_input.unsqueeze(0))
    memory_per_sample = torch.cuda.memory_allocated()  # Check memory allocation
    torch.cuda.empty_cache()  # Clear cache

    # Calculate the feasible batch size
    feasible_batch_size = target_memory_usage // memory_per_sample
    return feasible_batch_size


# Defining the UNet class, which inherits from nn.Module - the base class for all neural network modules in PyTorch.
class UNet(nn.Module):
    # The initializer method for the UNet class.
    def __init__(self, input_n=1024, in_channels=2, out_channels=1):
        print("input_n=", input_n)
        # Calling the initializer of the parent class (nn.Module).
        super(UNet, self).__init__()

        # Base number of channels, typically starts with 64
        base_channels = 64 # GPT-4 likes to suggest this :shrug:

        #base_channels = 128 # Use the doubler, Johnny

        #base_channels = 256 # let's see... out of vram womp-womp

        #base_channels = 196 # also runs out of memory

        #base_channels = 160 # maybe... it does train but very slowly

        #base_channels = max(16, input_n // 16)  # Doesn't work very well

        #base_channels = max(64, input_n // 8)  # 8 works with various LRs

        #base_channels = max(64, input_n // 4)  # works well with: lr=0.001 and ExponentialLR with gamma=0.9990 if
                                                # you're trying to overfit on purpose

        # Encoder Part: Consists of consecutive contracting blocks (convolution + pooling layers).

        # First contracting block of the encoder with the specified input channels and base output channels.
        self.encoder1 = self.contracting_block(in_channels, base_channels)
        # Second contracting block with doubled input channels and doubled output channels.
        self.encoder2 = self.contracting_block(base_channels, base_channels * 2)
        # Third contracting block with quadrupled input channels and quadrupled output channels.
        self.encoder3 = self.contracting_block(base_channels * 2, base_channels * 4)
        # Fourth contracting block with 8 times the input channels and 8 times the output channels.
        self.encoder4 = self.contracting_block(base_channels * 4, base_channels * 8)

        # Bottleneck Part: This is the layer at the bottom of the "U" shape, connecting encoder and decoder.
        # It further processes the feature map from the last encoder block.
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_channels * 8, base_channels * 16, 3, padding=1),  # Convolutional layer with 16 times the base channels.
            nn.ReLU(inplace=True),  # ReLU activation function.
            nn.Conv2d(base_channels * 16, base_channels * 16, 3, padding=1),  # Another convolutional layer.
            nn.ReLU(inplace=True)  # ReLU activation function.
        )

        # Decoder Part: Consists of expansive blocks (upsampling + convolution layers).

        # First expansive block of the decoder, taking in concatenated outputs from the bottleneck and the last encoder block.
        self.decoder1 = self.expansive_block(base_channels * 16 + base_channels * 8, base_channels * 8, base_channels * 4)
        # Second expansive block.
        self.decoder2 = self.expansive_block(base_channels * 4 + base_channels * 4, base_channels * 4, base_channels * 2)
        # Third expansive block.
        self.decoder3 = self.expansive_block(base_channels * 2 + base_channels * 2, base_channels * 2, base_channels)
        # Final layer to reduce the channel size to the desired number of output channels (e.g., 1 for binary segmentation).
        self.decoder4 = self.expansive_block(base_channels + base_channels, base_channels, base_channels)

        # Correcting the Final Layer: A 1x1 convolution to get the desired number of output channels
        self.final_layer = nn.Conv2d(base_channels, out_channels, 1)


    # Defining the contracting block, which is used in the encoder.
    def contracting_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),  # Convolutional layer.
            nn.ReLU(inplace=True),  # ReLU activation function.
            nn.Conv2d(out_channels, out_channels, 3, padding=1),  # Another convolutional layer.
            nn.ReLU(inplace=True),  # ReLU activation function.
            nn.MaxPool2d(kernel_size=2, stride=2)  # Max pooling layer for downsampling.
        )
        return block

    # Defining the expansive block, which is used in the decoder.
    def expansive_block(self, in_channels, mid_channel, out_channels):
        block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channel, 3, padding=1),  # Convolutional layer.
            nn.ReLU(inplace=True),  # ReLU activation function.
            nn.Conv2d(mid_channel, mid_channel, 3, padding=1),  # Another convolutional layer.
            nn.ReLU(inplace=True),  # ReLU activation function.
            nn.ConvTranspose2d(mid_channel, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)  # Transposed convolutional layer for upsampling.
        )
        return block

    # The forward method is overridden from nn.Module and defines the forward pass of the network.
    def forward(self, x):
        # Passing the input through the encoder and decoder, using skip connections.
        # Encoder:
        x1 = self.encoder1(x)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        x4 = self.encoder4(x3)

        # Bottleneck:
        x5 = self.bottleneck(x4)

        # Decoder (with skip connections):
        x = self.decoder1(torch.cat([x4, x5], 1))
        x = self.decoder2(torch.cat([x3, x], 1))
        x = self.decoder3(torch.cat([x2, x], 1))
        x = self.decoder4(torch.cat([x1, x], 1))
        x = self.final_layer(x)  # Pass through the final layer to get the correct output size
        return x


# Dataset class for loading the training images and ground truth masks
class UNetDataset(Dataset):
    """
    A custom Dataset class for loading the training images and ground truth masks.
    """
    def __init__(self, image_dir, transform=None):
        """
        Initializes the dataset object.
        :param image_dir: Directory where the training images are stored.
        :param transform: Transformations to be applied to the images.
        """
        print("******************************************************")
        self.image_dir = image_dir
        self.truth_dir = 'truth/'
        self.input_dir = 'input/'
        self.transform = transform
        #self.images = [file for file in os.listdir(image_dir + self.truth_dir)]
        #self.images.sort()
        print("image_dir + self.input_dir=", image_dir + self.input_dir)
        self.images = [file.split('_')[0] for file in os.listdir(image_dir + self.input_dir) if '_a.png' in file]
        #self.images.sort()
        shuffle(self.images)
        #self.images = self.images[0:3]
        print("files=", self.images)
        print("******************************************************")

    def __len__(self):
        """
        Returns the total number of image pairs in the dataset.
        """
        return len(self.images)

    def __getitem__(self, idx):
        """
        Retrieves an image pair (input and ground truth) from the dataset.
        :param idx: Index of the image pair to retrieve.
        :return: A tuple (input_image, ground_truth_mask).
        """
        base_filename = self.images[idx]

        img_a_path = os.path.join(self.image_dir + self.input_dir, base_filename + '_a.png')
        img_b_path = os.path.join(self.image_dir + self.input_dir, base_filename + '_b.png')
        mask_path = os.path.join(self.image_dir + self.truth_dir, base_filename + '.png')

        image_a = Image.open(img_a_path).convert("L")
        image_b = Image.open(img_b_path).convert("L")
        image_out = Image.open(mask_path).convert("L")

        if self.transform:
            image_a = self.transform(image_a)
            image_b = self.transform(image_b)
            #image_out = self.transform(image_out)
            image_out = transforms.functional.to_tensor(image_out).type(torch.float32)
        image = torch.cat([image_a, image_b], 0)  # Combining the two images into a 2-channel input
        return image, image_out

    def print_unique_mask_values(self):
        """
        Prints unique values present in the masks.
        """
        for img_name in self.images:
            mask_path = os.path.join(self.image_dir + self.truth_dir, img_name)
            mask = Image.open(mask_path).convert("L")
            mask_np = np.array(mask)
            unique_values = np.unique(mask_np)
            print(f"Unique values in mask {img_name}: {unique_values}")



class CustomHybridLoss(nn.Module):
    def __init__(self):
        super(CustomHybridLoss, self).__init__()

    def forward(self, outputs, targets):
        # Safeguard against invalid values
        outputs = torch.where(torch.isnan(outputs), torch.zeros_like(outputs), outputs)
        targets = torch.where(torch.isnan(targets), torch.zeros_like(targets), targets)

        # Calculate standard MSE Loss
        mse_loss = nn.functional.mse_loss(outputs, targets)

        # Apply Sobel filter to both outputs and targets
        sobel_outputs = self.sobel_filter(outputs)
        sobel_targets = self.sobel_filter(targets)

        # Calculate MSE Loss for Sobel filtered outputs and targets
        sobel_mse_loss = nn.functional.mse_loss(sobel_outputs, sobel_targets)

        # Combine the two losses
        #combined_loss = 0.5 * mse_loss + 0.5 * sobel_mse_loss
        combined_loss = sobel_mse_loss
        return combined_loss

    def sobel_filter(self, x):
        # Define the Sobel filter kernels
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=x.device)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=x.device)

        # Add batch and channel dimensions
        sobel_x = sobel_x.unsqueeze(0).unsqueeze(0)
        sobel_y = sobel_y.unsqueeze(0).unsqueeze(0)

        # Convolve with the input
        edge_x = nn.functional.conv2d(x, sobel_x, padding=1, groups=x.shape[1])
        edge_y = nn.functional.conv2d(x, sobel_y, padding=1, groups=x.shape[1])

        # Compute the magnitude of the gradients
        # Add a small epsilon to avoid sqrt of zero
        edge_mag = torch.sqrt(edge_x ** 2 + edge_y ** 2 + 1e-8)

        return edge_mag

def main():
    force_training = False
    model_file_path = "unet_model.pth"

    # Define transformations for the images
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Initialize the U-Net model
    model = UNet()

    model.to(device)

    # We'll load the training data in batches for better performance
    target_memory_usage = 1024 * 1024 * 1024 * 3  # For example, 3GB
    input_shape = (2, 1024, 1024)  # Example input shape, adjust as needed
    #batch_size = estimate_batch_size(model, input_shape, target_memory_usage)
    batch_size = 1 
    print("batch_size=", batch_size)

    # Creating dataset and dataloader for training
    train_dataset = UNetDataset("./unet_2_view_training/training/", transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Check if model file exists
    if os.path.exists(model_file_path) and not force_training:
        print(f"Loading model from {model_file_path}")
        model.load_state_dict(torch.load(model_file_path))
        model.eval()  # Set the model to evaluation mode

        # Creating dataset and dataloader for checking mask values -- debug
        #check_dataset = UNetDataset("./unetTrainingImages/", transform=transform)
        #check_dataset.print_unique_mask_values()  # Print unique values in the masks -- should be [  0 255]
    else:
        # Proceed with training
        #criterion = nn.BCEWithLogitsLoss()
        criterion = nn.MSELoss()
        #criterion = CustomHybridLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.00015)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.980)

        try:
            num_epochs = 200
            iteration = 0
            for epoch in range(num_epochs):
                for i, (inputs, masks) in enumerate(train_loader):
                    if select.select([sys.stdin], [], [], 0)[0]:
                        print("Early stopping initiated.")
                        raise StopIteration

                    inputs, masks = inputs.to(device), masks.to(device)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, masks)
                    loss.backward()
                    optimizer.step()

                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}, LR: {current_lr}")
                # Save model every 500 iterations
                iteration += 1
                if iteration % 500 == 0:
                    torch.save(model.state_dict(), model_file_path)
                    print(f"Model saved at iteration {iteration}")

                scheduler.step()

        except StopIteration:
            print("Training stopped early.")

        # Save the trained model
        torch.save(model.state_dict(), model_file_path)
        print("Final model saved.")

        model.eval()  # Set the model to evaluation mode

    # Inference on an unseen test image
    test_image_path = "./unet_2_view_training/test/"
    base_name = "00002"
    #base_name = "00121"

    #test_image_path = "unetTestImages/test_image.png"
    #test_image_path = "unetTestImages/test_image_orig.png"
    test_image_a = Image.open(test_image_path + base_name + "_a.png").convert("L")
    test_image_b = Image.open(test_image_path + base_name + "_b.png").convert("L")
    test_image_a = transform(test_image_a)
    test_image_b = transform(test_image_b)
    test_image = torch.cat([test_image_a, test_image_b], 0)  # Combining the two images into a 2-channel input
    test_image = test_image.unsqueeze(0)

    # Move test image to the GPU
    test_image = test_image.to(device)

    for i, (inputs, masks) in enumerate(train_loader):
        print("input image=", inputs)
        break
    print("test image=", test_image)

    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        prediction = model(test_image)
    '''
    # Apply threshold to the prediction to obtain binary masks
    predicted_mask = (prediction > 0).squeeze().cpu().numpy()  # Applying threshold

    # Process the prediction and save the output mask
    #output_mask_image = Image.fromarray((predicted_mask * 255).astype('uint8'))
    output_mask_image = Image.fromarray((predicted_mask * 65535).astype('uint16'))
    output_mask_image.save("predicted_mask.png")
    '''
    print("prediction=", prediction)
    predicted_mask = (prediction * 65535).squeeze().cpu().numpy()  # Applying threshold
    predicted_mask = np.clip(predicted_mask, 0, 65535)  # Ensuring values are within 16-bit range
    predicted_mask = predicted_mask.astype('uint16')  # Converting to 16-bit unsigned integer format

    # Save the 16-bit mask
    output_mask_image = Image.fromarray(predicted_mask)
    output_mask_image.save("predicted_mask.png")

    print("Inference complete. The output mask is saved as 'predicted_mask.png'.")

if __name__ == "__main__":
    main()
