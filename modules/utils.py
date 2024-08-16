import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torchvision.transforms.functional import InterpolationMode


def plot_images(X, batch=True, save=False, title='one'):
    """
    Plots a batch of images or a single image.

    Args:
        X (Tensor): Input tensor with shape [batch x 3 x h x w] for batch of images, or [3 x h x w] for a single image.
        batch (bool): If True, plots a batch of images in a grid. If False, plots a single image. Default is True.
        save (bool): If True, saves the plot as an image file. Default is False.
        title (str): Title for the saved image file. Default is 'one'.
    """
    if batch:
        # Number of columns in the grid
        num_cols = 8
        # Calculate number of rows needed
        num_rows = len(X) // num_cols if len(X) // num_cols else 1

        # Create a figure and a grid of subplots
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(num_cols, num_rows))

        # Plot each image in the grid
        for i in range(num_rows):
            for j in range(num_cols):
                idx = i * num_cols + j
                if idx < len(X):  # Check if index is within range
                    image = X[idx].permute(1, 2, 0).numpy()  # Convert to HxWxC format
                    axs[i, j].imshow((image * 255).astype("uint8"))
                    axs[i, j].axis('off')

        # Adjust spacing between subplots
        plt.subplots_adjust(wspace=0.025, hspace=0.025)

        # Save the figure if specified
        if save:
            plot_name = title + '.png'
            plt.savefig(plot_name)
    else:
        # Plot a single image
        plt.figure(figsize=(2, 2))
        plt.imshow(np.transpose(X, (1, 2, 0)))

    # Show the plot
    plt.show()

def compare_images(image1, image2, scale=1):
    """
    Compares two images side by side.

    Args:
        image1 (Tensor): First image tensor with shape [3 x h x w].
        image2 (Tensor): Second image tensor with shape [3 x h x w].
        scale (int): Scaling factor for the figure size. Default is 1.
    """
    # Convert images to HxWxC format
    image1 = np.transpose(image1.numpy(), (1, 2, 0))
    image2 = np.transpose(image2.numpy(), (1, 2, 0))

    # Create a new figure with a 1x2 grid of subplots
    fig, axes = plt.subplots(1, 2, figsize=(scale * 8, scale * 4))

    # Plot the first image
    axes[0].imshow(image1)
    axes[0].axis('off')
    axes[0].set_title('Initial Image')

    # Plot the second image
    axes[1].imshow(image2)
    axes[1].axis('off')
    axes[1].set_title('Transformed Image')

    # Adjust spacing between subplots
    plt.subplots_adjust(wspace=0.1)

    # Show the figure
    plt.show()

def augment_data(X, plot=False):
    """
    Applies data augmentation techniques to a batch of images.

    Args:
        X (Tensor): Input tensor with shape [batch x 3 x h x w].
        plot (bool): If True, plots images at each augmentation step. Default is False.

    Returns:
        Tensor: Augmented images.
    """
    if plot:
        print("########## ORIGINAL IMAGES ##########")
        plot_images(X)

    # Gamma correction
    gamma = 1.8
    X_gamma = torch.clamp(X * gamma, 0, 1)
    if plot:
        print("########## GAMMA IMAGES ##########")
        plot_images(X_gamma)

    # Sharpen the images
    X_sharper = TF.adjust_sharpness(X_gamma, sharpness_factor=3.5)
    if plot:
        print("########## SHARPER IMAGES ##########")
        plot_images(X_sharper)

    # Rotate the images +5 degrees
    X_rotated_plus = TF.rotate(X_sharper, angle=5, interpolation=InterpolationMode.BILINEAR)
    if plot:
        print("########## +5 ROTATED IMAGES ##########")
        plot_images(X_rotated_plus)

    # Rotate the images -5 degrees
    X_rotated_minus = TF.rotate(X_sharper, angle=-5, interpolation=InterpolationMode.BILINEAR)
    if plot:
        print("########## -5 ROTATED IMAGES ##########")
        plot_images(X_rotated_minus)

    # Concatenate images
    X_out = torch.cat((X_sharper, X_rotated_plus, X_rotated_minus), dim=0)

    # Flip the images
    X_flipped = TF.hflip(X_sharper)
    if plot:
        print("########## FLIPPED IMAGES ##########")
        plot_images(X_flipped)

    # Rotate the flipped images +5 degrees
    X_flipped_rotated_plus = TF.rotate(X_flipped, angle=5, interpolation=InterpolationMode.BILINEAR)
    if plot:
        print("########## FLIPPED +5 ROTATED IMAGES ##########")
        plot_images(X_flipped_rotated_plus)

    # Rotate the flipped images -5 degrees
    X_flipped_rotated_minus = TF.rotate(X_flipped, angle=-5, interpolation=InterpolationMode.BILINEAR)
    if plot:
        print("########## FLIPPED -5 ROTATED IMAGES ##########")
        plot_images(X_flipped_rotated_minus)

    # Concatenate all augmented images
    X_out = torch.cat((X_out, X_flipped, X_flipped_rotated_plus, X_flipped_rotated_minus), dim=0)

    return X_out

def seed_all(seed_val=42):
    """
    Sets the seed for various random number generators to ensure reproducibility.

    Args:
        seed_val (int): Seed value to set. Default is 42.
    """
    # Set the seed for PyTorch
    torch.manual_seed(seed_val)
    
    # Set the seed for the Python standard library
    random.seed(seed_val)
    
    # Set the seed for NumPy
    np.random.seed(seed_val)
    
    # Set the seed for CUDA (if available)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_val)
        torch.cuda.manual_seed_all(seed_val)
    
    # Configure additional settings for deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# custom weights initialization 
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)