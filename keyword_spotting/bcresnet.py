# BC-ResNet: Broadcasted Residual Learning for Efficient Keyword Spotting
# Implementation based on the paper by Kim et al., 2021
# https://arxiv.org/abs/2106.04140

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class SubSpectralNorm(nn.Module):
    """
    Sub-spectral normalization layer that applies normalization across frequency sub-groups.
    This helps in learning better frequency representations for audio spectrograms.
    """
    def __init__(self, num_features: int, spec_groups: int = 16, affine: str = "Sub", batch: bool = True, dim: int = 2):
        super().__init__()
        self.spec_groups = spec_groups
        self.affine_all = False
        affine_norm = False
        
        if affine == "Sub":  # affine transform for each sub group
            affine_norm = True
        elif affine == "All":
            self.affine_all = True
            self.weight = nn.Parameter(torch.ones((1, num_features, 1, 1)))
            self.bias = nn.Parameter(torch.zeros((1, num_features, 1, 1)))
            
        if batch:
            self.ssnorm = nn.BatchNorm2d(num_features * spec_groups, affine=affine_norm)
        else:
            self.ssnorm = nn.InstanceNorm2d(num_features * spec_groups, affine=affine_norm)
        
        self.sub_dim = dim

    def forward(self, x):
        """Apply sub-spectral normalization when dim h is frequency dimension"""
        if self.sub_dim in (3, -1):
            x = x.transpose(2, 3)
            x = x.contiguous()
            
        b, c, h, w = x.size()
        assert h % self.spec_groups == 0
        
        x = x.view(b, c * self.spec_groups, h // self.spec_groups, w)
        x = self.ssnorm(x)
        x = x.view(b, c, h, w)
        
        if self.affine_all:
            x = x * self.weight + self.bias
            
        if self.sub_dim in (3, -1):
            x = x.transpose(2, 3)
            x = x.contiguous()
            
        return x


class ConvBNReLU(nn.Module):
    """
    Basic convolution block with batch normalization and ReLU activation.
    Supports both 1D and 2D convolutions with various configurations.
    """
    def __init__(
        self,
        in_plane: int,
        out_plane: int,
        idx: int,
        kernel_size=3,
        stride=1,
        groups=1,
        use_dilation=False,
        activation=True,
        swish=False,
        BN=True,
        ssn=False,
    ):
        super().__init__()
        self.idx = idx

        def get_padding(kernel_size, use_dilation):
            rate = 1  # dilation rate
            padding_len = (kernel_size - 1) // 2
            if use_dilation and kernel_size > 1:
                rate = int(2**self.idx)
                padding_len = rate * padding_len
            return padding_len, rate

        # padding and dilation rate
        if isinstance(kernel_size, (list, tuple)):
            padding = []
            rate = []
            for k_size in kernel_size:
                temp_padding, temp_rate = get_padding(k_size, use_dilation)
                rate.append(temp_rate)
                padding.append(temp_padding)
        else:
            padding, rate = get_padding(kernel_size, use_dilation)

        # convbnrelu block
        layers = []
        layers.append(
            nn.Conv2d(in_plane, out_plane, kernel_size, stride, padding, rate, groups, bias=False)
        )
        
        if ssn:
            layers.append(SubSpectralNorm(out_plane, 5))
        elif BN:
            layers.append(nn.BatchNorm2d(out_plane))
            
        if swish:
            layers.append(nn.SiLU(True))
        elif activation:
            layers.append(nn.ReLU(True))
            
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class BCResBlock(nn.Module):
    """
    Broadcasted Residual Block - the core component of BC-ResNet.
    Uses 2D convolution for frequency processing and 1D convolution for temporal processing,
    connected via a broadcasted residual connection.
    """
    def __init__(self, in_plane: int, out_plane: int, idx: int, stride: Tuple[int, int]):
        super().__init__()
        self.transition_block = in_plane != out_plane
        kernel_size = (3, 3)

        # 2D part (f2) - processes frequency dimension
        layers = []
        if self.transition_block:
            layers.append(ConvBNReLU(in_plane, out_plane, idx, 1, 1))
            in_plane = out_plane
            
        layers.append(
            ConvBNReLU(
                in_plane,
                out_plane,
                idx,
                (kernel_size[0], 1),
                (stride[0], 1),
                groups=in_plane,
                ssn=True,
                activation=False,
            )
        )
        self.f2 = nn.Sequential(*layers)
        self.avg_gpool = nn.AdaptiveAvgPool2d((1, None))

        # 1D part (f1) - processes temporal dimension
        self.f1 = nn.Sequential(
            ConvBNReLU(
                out_plane,
                out_plane,
                idx,
                (1, kernel_size[1]),
                (1, stride[1]),
                groups=out_plane,
                swish=True,
                use_dilation=True,
            ),
            nn.Conv2d(out_plane, out_plane, 1, bias=False),
            nn.Dropout2d(0.1),
        )

    def forward(self, x):
        # 2D part
        shortcut = x
        x = self.f2(x)
        aux_2d_res = x
        x = self.avg_gpool(x)

        # 1D part
        x = self.f1(x)
        x = x + aux_2d_res  # broadcasted residual connection
        
        if not self.transition_block:
            x = x + shortcut
            
        x = F.relu(x, True)
        return x


def BCBlockStage(num_layers: int, last_channel: int, cur_channel: int, idx: int, use_stride: bool):
    """Create a stage of BC-ResBlocks"""
    stage = nn.ModuleList()
    channels = [last_channel] + [cur_channel] * num_layers
    
    for i in range(num_layers):
        stride = (2, 1) if use_stride and i == 0 else (1, 1)
        stage.append(BCResBlock(channels[i], channels[i + 1], idx, stride))
        
    return stage


class BCResNet(nn.Module):
    """
    BC-ResNet: Broadcasted Residual Network for Keyword Spotting
    
    Achieves state-of-the-art performance on Google Speech Commands dataset:
    - v1: 98.0% accuracy
    - v2: 98.7% accuracy
    
    Args:
        base_c: Base channel multiplier (controls model size)
        num_classes: Number of output classes (default: 12 for Speech Commands)
    """
    def __init__(self, base_c: int, num_classes: int = 12):
        super().__init__()
        self.num_classes = num_classes
        self.n = [2, 2, 4, 4]  # identical modules repeated n times
        self.c = [
            base_c * 2,
            base_c,
            int(base_c * 1.5),
            base_c * 2,
            int(base_c * 2.5),
            base_c * 4,
        ]  # num channels
        self.s = [1, 2]  # stage using stride
        self._build_network()

    def _build_network(self):
        # Head: (Conv-BN-ReLU)
        self.cnn_head = nn.Sequential(
            nn.Conv2d(1, self.c[0], 5, (2, 1), 2, bias=False),
            nn.BatchNorm2d(self.c[0]),
            nn.ReLU(True),
        )
        
        # Body: BC-ResBlocks
        self.BCBlocks = nn.ModuleList([])
        for idx, n in enumerate(self.n):
            use_stride = idx in self.s
            self.BCBlocks.append(BCBlockStage(n, self.c[idx], self.c[idx + 1], idx, use_stride))

        # Classifier
        self.classifier = nn.Sequential(
            nn.Conv2d(
                self.c[-2], self.c[-2], (5, 5), bias=False, groups=self.c[-2], padding=(0, 2)
            ),
            nn.Conv2d(self.c[-2], self.c[-1], 1, bias=False),
            nn.BatchNorm2d(self.c[-1]),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(self.c[-1], self.num_classes, 1),
        )

    def forward(self, x):
        x = self.cnn_head(x)
        
        for i, num_modules in enumerate(self.n):
            for j in range(num_modules):
                x = self.BCBlocks[i][j](x)
                
        x = self.classifier(x)
        x = x.view(-1, x.shape[1])
        return x


def create_bcresnet(tau: float = 1.0, num_classes: int = 12) -> BCResNet:
    """
    Create BC-ResNet model with specified scale factor.
    
    Args:
        tau: Scale factor for model size (1, 1.5, 2, 3, 6, 8)
        num_classes: Number of output classes
        
    Returns:
        BC-ResNet model
    """
    base_c = int(tau * 8)
    return BCResNet(base_c, num_classes)


if __name__ == "__main__":
    # Test model creation and forward pass
    model = create_bcresnet(tau=1.0)
    print(f"BC-ResNet-1.0 created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test with dummy input (batch_size=2, channels=1, freq=40, time=101)
    x = torch.randn(2, 1, 40, 101)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Output predictions: {torch.argmax(y, dim=1)}")
