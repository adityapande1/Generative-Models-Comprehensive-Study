import torch
import torch.nn as nn


class Generator(nn.Module):
    """
    Generator Network for a Generative Adversarial Network (GAN).

    Args:
        ngpu (int): Number of GPUs to use.
        nz (int): Size of the latent vector (input noise).
        ngf (int): Number of feature maps in the first layer.
        nc (int): Number of channels in the output image.
    """
    def __init__(self, ngpu=1, nz=100, ngf=64, nc=3):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. ``(ngf*8) x 4 x 4``
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. ``(ngf*4) x 8 x 8``
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. ``(ngf*2) x 16 x 16``
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. ``(ngf) x 32 x 32``
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. ``(nc) x 64 x 64``
        )

    def forward(self, input):
        """
        Forward pass through the generator network.

        Args:
            input (Tensor): Input tensor of shape (batch_size, nz, 1, 1).

        Returns:
            Tensor: Generated image tensor of shape (batch_size, nc, 64, 64).
        """
        return self.main(input)


class Discriminator(nn.Module):
    """
    Discriminator Network for a Generative Adversarial Network (GAN).

    Args:
        ngpu (int): Number of GPUs to use.
        nz (int): Size of the latent vector (input noise).
        ndf (int): Number of feature maps in the first layer.
        nc (int): Number of channels in the input image.
    """
    def __init__(self, ngpu=1, nz=100, ndf=64, nc=3):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is ``(nc) x 64 x 64``
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf) x 32 x 32``
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*2) x 16 x 16``
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*4) x 8 x 8``
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*8) x 4 x 4``
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        """
        Forward pass through the discriminator network.

        Args:
            input (Tensor): Input image tensor of shape (batch_size, nc, 64, 64).

        Returns:
            Tensor: Discriminator output tensor of shape (batch_size, 1, 1, 1).
        """
        return self.main(input)
