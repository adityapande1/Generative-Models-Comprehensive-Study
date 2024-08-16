import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    """
    Variational Autoencoder (VAE) implementation.
    
    Attributes:
        latent_dim (int): Dimension of the latent space.
        kld_w (float): Weight for the KLD loss term.
        encoder (nn.Sequential): Encoder network composed of convolutional layers.
        fc_mu (nn.Linear): Linear layer to output mean of the latent space.
        fc_var (nn.Linear): Linear layer to output log-variance of the latent space.
        decoder_input (nn.Linear): Linear layer to transform latent codes into the decoder input space.
        decoder (nn.Sequential): Decoder network composed of transposed convolutional layers.
        final_layer (nn.Sequential): Final layer to produce the reconstructed image.
    """
    
    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims=None,
                 kld_w=0.00025):
        """
        Initializes the VAE model.

        Args:
            in_channels (int): Number of input channels.
            latent_dim (int): Dimension of the latent space.
            hidden_dims (list[int], optional): List of dimensions for hidden layers in encoder/decoder.
            kld_w (float, optional): Weight for the KLD loss term. Default is 0.00025.
        """
        super(VAE, self).__init__()

        self.latent_dim = latent_dim
        self.kld_w = kld_w 

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)

        # Output dimensions of the encoder before linear layers
        self.fc_mu = nn.Linear(hidden_dims[-1]*64, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1]*64, latent_dim)

        # Build Decoder
        modules = []
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 64)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1],
                               hidden_dims[-1],
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels=3,
                      kernel_size=3, padding=1),
            nn.Tanh())
    
    def encode(self, x):
        """
        Encodes the input image into latent space.

        Args:
            x (Tensor): Input tensor with shape [N x C x H x W].

        Returns:
            List[Tensor]: List containing mean (mu) and log-variance (log_var) of the latent distribution.
        """
        result = self.encoder(x)
        result = torch.flatten(result, start_dim=1)

        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z):
        """
        Decodes the latent codes into the image space.

        Args:
            z (Tensor): Latent codes with shape [B x D].

        Returns:
            Tensor: Reconstructed image with shape [B x C x H x W].
        """
        result = self.decoder_input(z)
        result = result.view(-1, 128, 8, 8)
        
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu, logvar):
        """
        Applies the reparameterization trick to sample from the latent distribution.

        Args:
            mu (Tensor): Mean of the latent distribution with shape [B x D].
            logvar (Tensor): Log-variance of the latent distribution with shape [B x D].

        Returns:
            Tensor: Sampled latent codes with shape [B x D].
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x):
        """
        Performs a forward pass through the VAE model.

        Args:
            x (Tensor): Input tensor with shape [B x C x H x W].

        Returns:
            List[Tensor]: List containing reconstructed image, original image, mu, and log_var.
        """
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return (self.decode(z), x, mu, log_var)

    def loss_function(self, args):
        """
        Computes the VAE loss function, combining reconstruction loss and KLD loss.

        Args:
            args (List[Tensor]): List containing reconstructed image, original image, mu, and log_var.

        Returns:
            dict: Dictionary containing total loss, reconstruction loss, and KLD loss.
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]
        
        kld_weight = self.kld_w
        recons_loss = F.mse_loss(recons, input)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss': recons_loss.detach(), 'KLD': -kld_loss.detach()}

    def sample(self, num_samples, current_device):
        """
        Samples from the latent space and generates corresponding images.

        Args:
            num_samples (int): Number of samples to generate.
            current_device (int): Device to run the model on.

        Returns:
            Tensor: Generated images with shape [num_samples x C x H x W].
        """
        z = torch.randn(num_samples, self.latent_dim)
        z = z.to(current_device)
        samples = self.decode(z)
        return samples

    def generate(self, x):
        """
        Generates a reconstructed image from an input image.

        Args:
            x (Tensor): Input image with shape [B x C x H x W].

        Returns:
            Tensor: Reconstructed image with shape [B x C x H x W].
        """
        return self.forward(x)[0]
