"""
U-Net model for music source separation.

Based on:
- Jansson, A., et al. (2017). "Singing voice separation with deep u-net 
  convolutional networks." ISMIR 2017.
- Spleeter implementation from Deezer Research.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


class ConvBlock(nn.Module):
    """Convolutional block with batch normalization and activation."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int] = (5, 5),
        stride: Tuple[int, int] = (2, 2),
        padding: Tuple[int, int] = (2, 2),
        activation: str = "leaky_relu",
        use_batchnorm: bool = True,
    ):
        super().__init__()
        
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        
        self.use_batchnorm = use_batchnorm
        if use_batchnorm:
            self.bn = nn.BatchNorm2d(out_channels)
        
        if activation == "leaky_relu":
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif activation == "elu":
            self.activation = nn.ELU(inplace=True)
        else:
            self.activation = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.use_batchnorm:
            x = self.bn(x)
        x = self.activation(x)
        return x


class DeconvBlock(nn.Module):
    """Deconvolutional block with batch normalization and activation."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int] = (5, 5),
        stride: Tuple[int, int] = (2, 2),
        padding: Tuple[int, int] = (2, 2),
        output_padding: Tuple[int, int] = (1, 1),
        activation: str = "relu",
        use_batchnorm: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.deconv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        )
        
        self.use_batchnorm = use_batchnorm
        if use_batchnorm:
            self.bn = nn.BatchNorm2d(out_channels)
        
        if activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif activation == "leaky_relu":
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == "elu":
            self.activation = nn.ELU(inplace=True)
        else:
            self.activation = nn.Identity()
        
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.deconv(x)
        if self.use_batchnorm:
            x = self.bn(x)
        x = self.activation(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x


class UNet(nn.Module):
    """
    U-Net architecture for music source separation.
    
    The model takes a mix spectrogram as input and outputs masks for each source.
    Architecture follows the paper by Jansson et al. (2017) with 6 encoder 
    and 6 decoder layers.
    
    Args:
        n_fft: FFT size for STFT (default: 4096)
        n_sources: Number of output sources (default: 4)
        n_channels: Number of input channels (default: 2 for stereo)
        conv_filters: List of filter sizes for encoder layers
    """
    
    def __init__(
        self,
        n_fft: int = 4096,
        n_sources: int = 4,
        n_channels: int = 2,
        conv_filters: Optional[List[int]] = None,
    ):
        super().__init__()
        
        self.n_fft = n_fft
        self.n_sources = n_sources
        self.n_channels = n_channels
        
        # Default filter sizes following Spleeter configuration
        if conv_filters is None:
            conv_filters = [16, 32, 64, 128, 256, 512]
        
        self.conv_filters = conv_filters
        
        # Calculate frequency bins (n_fft // 2 + 1)
        self.n_bins = n_fft // 2 + 1
        
        # Encoder (downsampling path)
        self.encoder_layers = nn.ModuleList()
        in_ch = n_channels
        
        for i, out_ch in enumerate(conv_filters):
            self.encoder_layers.append(
                ConvBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    activation="leaky_relu",
                )
            )
            in_ch = out_ch
        
        # Decoder (upsampling path)
        self.decoder_layers = nn.ModuleList()
        conv_filters_rev = list(reversed(conv_filters[:-1]))
        
        for i, out_ch in enumerate(conv_filters_rev):
            # Input channels = current + skip connection
            in_ch = conv_filters[-(i+1)] + conv_filters[-(i+2)]
            
            # Add dropout to the first few decoder layers
            dropout = 0.5 if i < 3 else 0.0
            
            self.decoder_layers.append(
                DeconvBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    activation="relu",
                    dropout=dropout,
                )
            )
        
        # Final layer to reconstruct masks for each source
        self.final_conv = nn.Conv2d(
            in_channels=conv_filters[0] + n_channels,  # + skip connection from input
            out_channels=n_channels * n_sources,
            kernel_size=(4, 4),
            padding=(2, 2),
            dilation=(2, 2),
        )
        
        # Sigmoid activation for masks
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through U-Net.
        
        Args:
            x: Input spectrogram tensor of shape (B, C, F, T)
               where B=batch, C=channels, F=frequency bins, T=time frames
        
        Returns:
            masks: Output masks tensor of shape (B, C*S, F, T)
                   where S=number of sources
        """
        # Store skip connections
        skip_connections = [x]
        
        # Encoder path
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)
            skip_connections.append(x)
        
        # Remove the last skip connection (it's the bottleneck)
        skip_connections = skip_connections[:-1]
        
        # Decoder path
        for i, decoder_layer in enumerate(self.decoder_layers):
            # Concatenate with corresponding skip connection
            skip = skip_connections[-(i+1)]
            
            # Ensure skip connection matches current tensor size
            if skip.shape[2:] != x.shape[2:]:
                # Use adaptive pooling to match sizes
                skip = torch.nn.functional.adaptive_avg_pool2d(skip, x.shape[2:])
            
            x = torch.cat([x, skip], dim=1)
            x = decoder_layer(x)
        
        # Final layer with input skip connection
        final_skip = skip_connections[0]
        if final_skip.shape[2:] != x.shape[2:]:
            final_skip = torch.nn.functional.adaptive_avg_pool2d(final_skip, x.shape[2:])
        
        x = torch.cat([x, final_skip], dim=1)
        x = self.final_conv(x)
        
        # Apply sigmoid to get masks between 0 and 1
        masks = self.sigmoid(x)
        
        # Output shape: (B, C*S, F, T) where C*S = channels * sources
        # This matches the expected format for source separation
        return masks


class SpleeterUNet(UNet):
    """
    U-Net implementation following Spleeter specifications.
    
    This variant uses the exact configuration from the Spleeter paper
    with specific filter sizes and architectural choices.
    """
    
    def __init__(
        self,
        n_fft: int = 4096,
        n_sources: int = 4,
        n_channels: int = 2,
    ):
        # Spleeter uses these specific filter sizes
        conv_filters = [16, 32, 64, 128, 256, 512]
        
        super().__init__(
            n_fft=n_fft,
            n_sources=n_sources,
            n_channels=n_channels,
            conv_filters=conv_filters,
        )


def create_unet_model(
    model_type: str = "unet",
    n_fft: int = 4096,
    n_sources: int = 4,
    n_channels: int = 2,
    **kwargs
) -> nn.Module:
    """
    Factory function to create U-Net models.
    
    Args:
        model_type: Type of model ("unet" or "spleeter")
        n_fft: FFT size for STFT
        n_sources: Number of output sources
        n_channels: Number of input channels
        **kwargs: Additional arguments for model construction
    
    Returns:
        U-Net model instance
    """
    if model_type == "spleeter":
        return SpleeterUNet(
            n_fft=n_fft,
            n_sources=n_sources,
            n_channels=n_channels,
        )
    elif model_type == "unet":
        return UNet(
            n_fft=n_fft,
            n_sources=n_sources,
            n_channels=n_channels,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Test the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model
    model = create_unet_model("spleeter").to(device)
    
    # Test input (batch_size=2, channels=2, freq_bins=2049, time_frames=256)
    test_input = torch.randn(2, 2, 2049, 256).to(device)
    
    print(f"Model: {model.__class__.__name__}")
    print(f"Input shape: {test_input.shape}")
    
    # Forward pass
    with torch.no_grad():
        output = model(test_input)
    
    print(f"Output shape: {output.shape}")
    print(f"Expected shape: (batch_size, channels, freq_bins, time_frames, sources)")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
