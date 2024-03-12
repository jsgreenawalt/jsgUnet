# jsgUnet
A U-Net implementation in PyTorch

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
