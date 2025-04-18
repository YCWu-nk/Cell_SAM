"""
The network architectures is based on the implementation of CycleGAN and CUT
Original PyTorch repo of CycleGAN: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
Original PyTorch repo of CUT: https://github.com/taesungp/contrastive-unpaired-translation
Original CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
Original CUT paper: https://arxiv.org/pdf/2007.15651.pdf
We use the network architecture for our default modal image translation
"""

import torch.nn.functional as F
import torch.autograd
import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
import numpy as np
from torch.nn import init
import math 

class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)

def get_timestep_embedding(timesteps, embedding_dim, max_positions=10000):
    assert len(timesteps.shape) == 1  # and timesteps.dtype == tf.int32
    half_dim = embedding_dim // 2
      # magic number 10000 is from transformers
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1), mode='constant')
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb

##################################################################################
# Discriminator
##################################################################################
class D_NLayersMulti(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3,
                 norm_layer=nn.BatchNorm2d, num_D=1):
        super(D_NLayersMulti, self).__init__()
        # st()
        self.num_D = num_D
        if num_D == 1:
            layers = self.get_layers(input_nc, ndf, n_layers, norm_layer)
            self.model = nn.Sequential(*layers)
        else:
            layers = self.get_layers(input_nc, ndf, n_layers, norm_layer)
            self.add_module("model_0", nn.Sequential(*layers))
            self.down = nn.AvgPool2d(3, stride=2, padding=[
                                     1, 1], count_include_pad=False)
            for i in range(1, num_D):
                ndf_i = int(round(ndf / (2**i)))
                layers = self.get_layers(input_nc, ndf_i, n_layers, norm_layer)
                self.add_module("model_%d" % i, nn.Sequential(*layers))

    def get_layers(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw,
                              stride=2, padding=padw), nn.LeakyReLU(0.2, True)]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1,
                               kernel_size=kw, stride=1, padding=padw)]

        return sequence

    def forward(self, input):
        if self.num_D == 1:
            return self.model(input)
        result = []
        down = input
        for i in range(self.num_D):
            model = getattr(self, "model_%d" % i)
            result.append(model(down))
            if i != self.num_D - 1:
                down = self.down(down)
        return result



class ConvBlock_cond(nn.Module):
    def __init__(self, in_channel, out_channel,t_emb_dim, kernel_size=4,stride=1,padding=1,norm_layer=None,downsample=True,use_bias=None):
        super().__init__()
        self.downsample=downsample
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding, bias=use_bias)
        
        if norm_layer is not None:
            self.use_norm =True
            self.norm = norm_layer(out_channel)
        else:
            self.use_norm = False
        self.act = nn.LeakyReLU(0.2, True)
        self.down = Downsample(out_channel)
        
        self.dense= nn.Linear(t_emb_dim, out_channel)
    def forward(self, input,t_emb):
        out = self.conv1(input)
        out += self.dense(t_emb)[..., None, None]
        if self.use_norm:
            out = self.norm(out)
        out = self.act(out)
        if self.downsample:
            out = self.down(out)
        
        return out
    
class NLayerDiscriminator_ncsn(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, no_antialias=False):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator_ncsn, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        self.model_main = nn.ModuleList()
        kw = 4
        padw = 1
        if no_antialias:
            sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        else:
            self.model_main.append(ConvBlock_cond(input_nc, ndf, 4*ndf,kernel_size=kw, stride=1, padding=padw,use_bias=use_bias))
        
        nf_mult = 1
        nf_mult_prev = 1
         
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            if no_antialias:
                sequence += [
                    nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                    norm_layer(ndf * nf_mult),
                    nn.LeakyReLU(0.2, True)]
            else:
                self.model_main.append(
                    ConvBlock_cond(ndf * nf_mult_prev, ndf * nf_mult, 4*ndf,kernel_size=kw, stride=1, padding=padw,use_bias=use_bias,norm_layer=norm_layer)
                    
                )

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        self.model_main.append(
            ConvBlock_cond(ndf * nf_mult_prev, ndf * nf_mult,4*ndf, kernel_size=kw, stride=1, padding=padw,use_bias=use_bias,norm_layer=norm_layer,downsample=False)
            
        )
        self.final_conv =nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
        self.t_embed = TimestepEmbedding(
            embedding_dim=4*ndf,
            hidden_dim=4*ndf,
            output_dim=4*ndf,
            act=nn.LeakyReLU(0.2),
        )

    def forward(self, input,t_emb,input2=None):
        """Standard forward."""
        t_emb = self.t_embed(t_emb)
        if input2 is not None:
            out = torch.cat([input,input2],dim=1)
        else:
            
            out = input
        for layer in self.model_main:
            out = layer(out,t_emb)
            
        return self.final_conv(out)
    
class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)


##################################################################################
# Generator
##################################################################################

class TimestepEmbedding(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim, act=nn.LeakyReLU(0.2)):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.main = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, output_dim),
            nn.LeakyReLU(0.2),
            # EqualLinear(hidden_dim, output_dim,bias_init = 0, activation='fused_lrelu')
        )

    def forward(self, temp):
        temb = get_timestep_embedding(temp, self.embedding_dim)
        temb = self.main(temb)
        return temb

def get_timestep_embedding(timesteps, embedding_dim, max_positions=10000):
    assert len(timesteps.shape) == 1  # and timesteps.dtype == tf.int32
    half_dim = embedding_dim // 2
      # magic number 10000 is from transformers
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1), mode='constant')
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb
    
class AdaptiveLayer(nn.Module):
    def __init__(self, in_channel, style_dim):
        super().__init__()

        self.style_net = nn.Linear(style_dim, in_channel * 2)

        self.style_net.bias.data[:in_channel] = 1
        self.style_net.bias.data[in_channel:] = 0

    def forward(self, input, style):
        
        style = self.style_net(style).unsqueeze(2).unsqueeze(3)
        gamma, beta = style.chunk(2, 1)

        out = gamma * input + beta

        return out


class TAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, kernel_size, num_channels=1, reduction_ratio=0.0625, num_kernels=4, min_channels=16):
        """
        Initialize the TAttention module.

        Args:
        embed_dim (int): The embedding dimension of the input features.
        num_heads (int): The number of attention heads.
        kernel_size (int): The size of the convolution kernel.
        num_channels (int, optional): The number of channels of the input features, default is 1.
        reduction_ratio (float, optional): The channel reduction ratio, default is 0.0625.
        num_kernels (int, optional): The number of convolution kernels, default is 4.
        min_channels (int, optional): The minimum number of channels after reduction, default is 16.
        """
        # Call the constructor of the parent class nn.Module
        super(TAttention, self).__init__()
        # Calculate the reduced number of channels, taking the maximum of the product of the embedding dimension and the reduction ratio and the minimum number of channels
        reduced_channels = max(int(embed_dim * reduction_ratio), min_channels)
        # Save the convolution kernel size
        self.kernel_size = kernel_size
        # Save the number of convolution kernels
        self.num_kernels = num_kernels
        # Initialize the temperature parameter, which is used to control the smoothness of the attention distribution
        self.temperature = 1.0

        # Global average pooling layer, which averages each channel of the input feature map to a single value
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        # Linear projection layer, which reduces the number of channels of the input features to reduced_channels
        self.linear_projection = nn.Conv2d(embed_dim, reduced_channels, 1, bias=False)
        # Batch normalization layer, which normalizes the reduced features
        self.batch_norm = nn.BatchNorm2d(reduced_channels)
        # ReLU activation function, which adds non - linearity to the model
        self.activation = nn.ReLU(inplace=True)

        # Channel projection layer, which restores the number of channels of the reduced features to embed_dim
        self.channel_projection = nn.Conv2d(reduced_channels, embed_dim, 1, bias=True)
        # Channel attention calculation function
        self.ch_attention_func = self.compute_ch_attention

        # If the embedding dimension is equal to the number of channels and the number of attention heads, use the identity function as the filter attention calculation function
        if embed_dim == num_channels and embed_dim == num_heads:  
            self.filter_attention_func = self.identity
        else:
            # Filter projection layer, which projects the number of channels of the reduced features to num_heads
            self.filter_projection = nn.Conv2d(reduced_channels, num_heads, 1, bias=True)
            # Filter attention calculation function
            self.filter_attention_func = self.compute_filter_attention

        # If the convolution kernel size is 1, use the identity function as the spatial attention calculation function
        if kernel_size == 1:  
            self.s_attention_func = self.identity
        else:
            # Spatial projection layer, which projects the number of channels of the reduced features to kernel_size * kernel_size
            self.spatial_projection = nn.Conv2d(reduced_channels, kernel_size * kernel_size, 1, bias=True)
            # Spatial attention calculation function
            self.s_attention_func = self.compute_s_attention

        # If the number of convolution kernels is 1, use the identity function as the kernel attention calculation function
        if num_kernels == 1:
            self.kernel_attention_func = self.identity
        else:
            # Kernel projection layer, which projects the number of channels of the reduced features to num_kernels
            self.kernel_projection = nn.Conv2d(reduced_channels, num_kernels, 1, bias=True)
            # Kernel attention calculation function
            self.kernel_attention_func = self.compute_kernel_attention

        # Initialize the weights of the model
        self.initialize_weights()

    def initialize_weights(self):
        """
        Initialize the weights of all convolutional layers and batch normalization layers in the model.
        """
        for module in self.modules():
            # If the module is a convolutional layer
            if isinstance(module, nn.Conv2d):
                # Initialize the weights of the convolutional layer using the Kaiming initialization method
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                # If the convolutional layer has a bias term, initialize it to 0
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            # If the module is a batch normalization layer
            if isinstance(module, nn.BatchNorm2d):
                # Initialize the weights of the batch normalization layer to 1
                nn.init.constant_(module.weight, 1)
                # Initialize the bias term of the batch normalization layer to 0
                nn.init.constant_(module.bias, 0)

    def update_temperature(self, temperature):
        """
        Update the temperature parameter.

        Args:
        temperature (float): The new temperature value.
        """
        self.temperature = temperature

    @staticmethod
    def identity(_):
        """
        Identity function, which returns 1.0.

        Args:
        _: The input parameter, which is not used here.

        Returns:
        float: 1.0
        """
        return 1.0

    def compute_ch_attention(self, x):
        """
        Compute the channel attention.

        Args:
        x (torch.Tensor): The input features.

        Returns:
        torch.Tensor: The channel attention weights.
        """
        # Pass through the channel projection layer, reshape the output, normalize it using the sigmoid function, and divide by the temperature parameter
        ch_attention = torch.sigmoid(self.channel_projection(x).view(x.size(0), -1, 1, 1) / self.temperature)
        return ch_attention

    def compute_filter_attention(self, x):
        """
        Compute the filter attention.

        Args:
        x (torch.Tensor): The input features.

        Returns:
        torch.Tensor: The filter attention weights.
        """
        # Pass through the filter projection layer, reshape the output, normalize it using the sigmoid function, and divide by the temperature parameter
        filter_attention = torch.sigmoid(self.filter_projection(x).view(x.size(0), -1, 1, 1) / self.temperature)
        return filter_attention

    def compute_s_attention(self, x):
        """
        Compute the spatial attention.

        Args:
        x (torch.Tensor): The input features.

        Returns:
        torch.Tensor: The spatial attention weights.
        """
        # Pass through the spatial projection layer, reshape the output, normalize it using the sigmoid function, and divide by the temperature parameter
        s_attention = self.spatial_projection(x).view(x.size(0), 1, 1, 1, self.kernel_size, self.kernel_size)
        s_attention = torch.sigmoid(s_attention / self.temperature)
        return s_attention

    def compute_kernel_attention(self, x):
        """
        Compute the kernel attention.

        Args:
        x (torch.Tensor): The input features.

        Returns:
        torch.Tensor: The kernel attention weights.
        """
        # Pass through the kernel projection layer, reshape the output, normalize it using the softmax function, and divide by the temperature parameter
        kernel_attention = self.kernel_projection(x).view(x.size(0), -1, 1, 1, 1, 1)
        kernel_attention = F.softmax(kernel_attention / self.temperature, dim=1)
        return kernel_attention

    def forward(self, x):
        """
        Forward propagation function.

        Args:
        x (torch.Tensor): The input features.

        Returns:
        tuple: A tuple containing the channel attention, filter attention, spatial attention, and kernel attention.
        """
        # Global average pooling
        x = self.global_avg_pool(x)
        # Linear projection
        x = self.linear_projection(x)
        # ReLU activation
        x = self.activation(x)
        # Compute the channel attention, filter attention, spatial attention, and kernel attention respectively
        return self.ch_attention_func(x), self.filter_attention_func(x), self.s_attention_func(x), self.kernel_attention_func(x)


import torch
import torch.nn as nn
import torch.nn.functional as F


class TConv2d(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, stride=1, padding=0, dilation=1, num_heads=1,
                 reduction_ratio=0.0625, num_kernels=4):
        """
        Initialize the TConv2d module.

        Args:
            in_dim (int): The number of input channels.
            out_dim (int): The number of output channels.
            kernel_size (int): The size of the convolution kernel.
            stride (int, optional): The stride of the convolution operation. Defaults to 1.
            padding (int, optional): The padding added to the input. Defaults to 0.
            dilation (int, optional): The spacing between kernel points. Defaults to 1.
            num_heads (int, optional): The number of attention heads. Defaults to 1.
            reduction_ratio (float, optional): The ratio for channel reduction. Defaults to 0.0625.
            num_kernels (int, optional): The number of convolution kernels. Defaults to 4.
        """
        # Call the constructor of the parent class nn.Module
        super(TConv2d, self).__init__()
        # Store the number of input channels
        self.in_dim = in_dim
        # Store the number of output channels
        self.out_dim = out_dim
        # Store the size of the convolution kernel
        self.kernel_size = kernel_size
        # Store the stride of the convolution
        self.stride = stride
        # Store the padding of the convolution
        self.padding = padding
        # Store the dilation of the convolution
        self.dilation = dilation
        # Store the number of attention heads
        self.num_heads = num_heads
        # Store the number of convolution kernels
        self.num_kernels = num_kernels
        # Initialize the attention layer
        self.attention_layer = TAttention(in_dim, out_dim, kernel_size, num_channels=num_heads,
                                          reduction_ratio=reduction_ratio, num_kernels=num_kernels)
        # Initialize the learnable weights as a parameter
        self.weight = nn.Parameter(torch.randn(num_kernels, out_dim, in_dim // num_heads, kernel_size, kernel_size),
                                   requires_grad=True)
        # Initialize the weights of the module
        self.initialize_weights()

        # Determine which forward implementation to use based on kernel size and number of kernels
        if self.kernel_size == 1 and self.num_kernels == 1:
            self.forward_impl = self.forward_impl_pointwise
        else:
            self.forward_impl = self.forward_impl_common

    def initialize_weights(self):
        """
        Initialize the weights of the convolution kernels.
        """
        for i in range(self.num_kernels):
            # Initialize the weights using Kaiming normal initialization
            nn.init.kaiming_normal_(self.weight[i], mode='fan_out', nonlinearity='relu')

    def update_temperature(self, temperature):
        """
        Update the temperature parameter of the attention layer.

        Args:
            temperature (float): The new temperature value.
        """
        self.attention_layer.update_temperature(temperature)

    def forward_impl_common(self, x):
        """
        The common forward implementation for non - pointwise convolutions.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after convolution and attention application.
        """
        # Compute attention maps from the attention layer
        ch_attention, filter_attention, s_attention, kernel_attention = self.attention_layer(x)
        # Get the shape of the input tensor
        batch_size, in_dim, height, width = x.size()
        # Apply channel attention to the input
        x = x * ch_attention
        # Reshape the input for grouped convolution
        x = x.reshape(1, -1, height, width)
        # Combine spatial attention, kernel attention, and the learnable weights
        a_weight = s_attention * kernel_attention * self.weight.unsqueeze(dim=0)
        # Sum over the kernel dimension and reshape the weights
        a_weight = torch.sum(a_weight, dim=1).view(
            [-1, self.in_dim // self.num_heads, self.kernel_size, self.kernel_size])
        # Perform the convolution operation
        output = F.conv2d(x, weight=a_weight, bias=None, stride=self.stride, padding=self.padding,
                          dilation=self.dilation, groups=self.num_heads * batch_size)
        # Reshape the output back to the original batch size
        output = output.view(batch_size, self.out_dim, output.size(-2), output.size(-1))
        # Apply filter attention to the output
        output = output * filter_attention
        return output

    def forward_impl_pointwise(self, x):
        """
        The forward implementation for pointwise convolutions (kernel size = 1 and num_kernels = 1).

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after convolution and attention application.
        """
        # Compute attention maps from the attention layer
        ch_attention, filter_attention, s_attention, kernel_attention = self.attention_layer(x)
        # Apply channel attention to the input
        x = x * ch_attention
        # Perform the pointwise convolution
        output = F.conv2d(x, weight=self.weight.squeeze(dim=0), bias=None, stride=self.stride, padding=self.padding,
                          dilation=self.dilation, groups=self.num_heads)
        # Apply filter attention to the output
        output = output * filter_attention
        return output

    def forward(self, x):
        """
        The forward pass of the TConv2d module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        return self.forward_impl(x)


class ResnetGenerator_ncsn(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=9,
                 padding_type='reflect',  no_antialias=False, no_antialias_up=False, opt=None):

        assert(n_blocks >= 0)
        super(ResnetGenerator_ncsn, self).__init__()
        self.opt = opt
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]
        self.ngf = ngf
        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            if no_antialias:
                model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                          norm_layer(ngf * mult * 2),
                          nn.ReLU(True)]
            else:
                model += [TConv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=1, padding=1),  ###
                          norm_layer(ngf * mult * 2),
                          nn.ReLU(True),
                          Downsample(ngf * mult * 2)
                          # nn.AvgPool2d(kernel_size=2, stride=2)
                        ]
        self.model_res = nn.ModuleList()
        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks
        
            self.model_res += [TransformerConvBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias,temb_dim=4*ngf,z_dim=4*ngf)]

        model_upsample = []
        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            if no_antialias_up:
                model_upsample += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
                          norm_layer(int(ngf * mult / 2)),
                          nn.ReLU(True)]
            else:
                model_upsample += [
                    Upsample(ngf * mult),
                    # nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                    TConv2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=1, padding=1), ####bias=use_bias),
                    norm_layer(int(ngf * mult / 2)),
                    nn.ReLU(True)]
        model_upsample += [nn.ReflectionPad2d(3)]
        model_upsample += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model_upsample += [nn.Tanh()]

        self.model = nn.Sequential(*model)
        self.model_upsample = nn.Sequential(*model_upsample)
        mapping_layers = [PixelNorm(),
                      nn.Linear(self.ngf*4, self.ngf*4),
                      nn.LeakyReLU(0.2)]
        for _ in range(opt.n_mlp):
            mapping_layers.append(nn.Linear(self.ngf*4, self.ngf*4))
            mapping_layers.append(nn.LeakyReLU(0.2))
        self.z_transform = nn.Sequential(*mapping_layers)
        modules_emb = []
        modules_emb += [nn.Linear(self.ngf,self.ngf*4)]

        nn.init.zeros_(modules_emb[-1].bias)
        modules_emb += [nn.LeakyReLU(0.2)]
        modules_emb += [nn.Linear(self.ngf*4,self.ngf*4)]

        nn.init.zeros_(modules_emb[-1].bias)
        modules_emb += [nn.LeakyReLU(0.2)]
        self.time_embed = nn.Sequential(*modules_emb)

    def forward(self, x, time_cond,z,layers=[], encode_only=False):
        z_embed = self.z_transform(z)
        # print(z_embed.shape)
        temb = get_timestep_embedding(time_cond, self.ngf)
        time_embed = self.time_embed(temb)
        if len(layers) > 0:
            feat = x
            feats = []
            for layer_id, layer in enumerate(self.model):
                feat = layer(feat)
                if layer_id in layers:
                    feats.append(feat)

            for layer_id, layer in enumerate(self.model_res):
                feat = layer(feat,time_embed,z_embed)
                if layer_id+len(self.model) in layers:
                    feats.append(feat)
                if layer_id+len(self.model) == layers[-1] and encode_only:
                    return feats
            return feat, feats
        else:

            out = self.model(x)
            for layer in self.model_res:
                out = layer(out,time_embed,z_embed)
            out = self.model_upsample(out)
            return out
##################################################################################
# Basic Blocks
##################################################################################
class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out
        
class TFer(nn.Module):

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias, temb_dim, z_dim, num_heads=4, num_layers=2):

        super(TFer, self).__init__()

        self.conv_block, self.adaptive, self.conv_fin = self.build_transformer_block(dim, padding_type, norm_layer, use_dropout, use_bias, temb_dim, z_dim, num_heads, num_layers)

    def build_transformer_block(self, dim, padding_type, norm_layer, use_dropout, use_bias, temb_dim, z_dim, num_heads, num_layers):

        
        self.conv_block = nn.ModuleList()
        self.conv_fin = nn.ModuleList()

        # Adaptive Layer (will remain as is)
        self.adaptive = AdaptiveLayer(dim, z_dim) 
        
        # Time conditioning
        self.Dense_time = nn.Linear(temb_dim, dim)
        nn.init.zeros_(self.Dense_time.bias)

        # Style conditioning
        self.style = nn.Linear(z_dim, dim * 2)
        self.style.bias.data[:dim] = 1
        self.style.bias.data[dim:] = 0

        # Transformer Block
        self.transformer_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.transformer_layers.append(
                nn.TransformerEncoderLayer(
                    d_model=dim, 
                    nhead=num_heads, 
                    dim_feedforward=dim * 4,  
                    dropout=0.1
                )
            )

        if use_dropout:
            self.conv_fin += [nn.Dropout(0.5)]

        return self.conv_block, self.adaptive, self.conv_fin

    def forward(self, x, time_cond, z):

        time_input = self.Dense_time(time_cond)
        
        batch_size, channels, height, width = x.size()

        x_flattened = x.view(batch_size, channels, -1).permute(0, 2, 1)  

        time_input = time_input.unsqueeze(1).expand(-1, height * width, -1)  
        x_flattened += time_input  

        for layer in self.transformer_layers:
            x_flattened = layer(x_flattened)

        x_transformed = x_flattened.permute(0, 2, 1).view(batch_size, channels, height, width)

        x_transformed = self.adaptive(x_transformed, z)

        for layer in self.conv_fin:
            x_transformed = layer(x_transformed)

        out = x + x_transformed

        return out


class TransformerConvBlock(nn.Module):
    """Define a block that combines Transformer and Convolution"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias, temb_dim, z_dim, num_heads=4):
        """
        Initialize the block.

        Args:
            dim (int): The number of channels in the input and output features.
            padding_type (str): The type of padding to use ('reflect', 'replicate', or 'zero').
            norm_layer (nn.Module): The normalization layer to use.
            use_dropout (bool): Whether to use dropout in the convolutional block.
            use_bias (bool): Whether to use bias in the convolutional layers.
            temb_dim (int): The dimension of the time conditioning vector.
            z_dim (int): The dimension of the style vector.
            num_heads (int, optional): The number of heads in the Transformer encoder layer. Defaults to 4.
        """
        # Call the constructor of the parent class
        super(TransformerConvBlock, self).__init__()
        # Build the convolutional block, adaptive layer, and final convolutional layers
        self.conv_block, self.adaptive, self.conv_fin = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias, temb_dim, z_dim)
        # Initialize a Transformer encoder layer
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=num_heads)
        # A linear layer to process the time conditioning vector
        self.Dense_time = nn.Linear(temb_dim, dim)
        # Initialize the bias of the time linear layer to zero
        nn.init.zeros_(self.Dense_time.bias)
        # A linear layer to generate style parameters from the style vector
        self.style = nn.Linear(z_dim, dim * 2)
        # Initialize the first half of the bias of the style layer to 1 and the second half to 0
        self.style.bias.data[:dim] = 1
        self.style.bias.data[dim:] = 0

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias, temb_dim, z_dim):
        """
        Construct a convolutional block.

        Args:
            dim (int): The number of channels in the input and output features.
            padding_type (str): The type of padding to use ('reflect', 'replicate', or 'zero').
            norm_layer (nn.Module): The normalization layer to use.
            use_dropout (bool): Whether to use dropout in the convolutional block.
            use_bias (bool): Whether to use bias in the convolutional layers.
            temb_dim (int): The dimension of the time conditioning vector.
            z_dim (int): The dimension of the style vector.

        Returns:
            tuple: A tuple containing the convolutional block, the adaptive layer, and the final convolutional layers.
        """
        # Initialize a module list for the convolutional block
        conv_block = nn.ModuleList()
        # Initialize a module list for the final convolutional layers
        conv_fin = nn.ModuleList()
        # Padding value
        p = 0
        # Add padding layer based on the padding type
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError(f'padding [{padding_type}] is not implemented')

        # Add a convolutional layer and a normalization layer
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]
        # Initialize the adaptive layer
        adaptive = AdaptiveLayer(dim, z_dim)
        # Add a ReLU activation layer
        conv_fin += [nn.ReLU(True)]
        # Add a dropout layer if specified
        if use_dropout:
            conv_fin += [nn.Dropout(0.5)]

        # Padding value for the second part
        p = 0
        # Add padding layer based on the padding type for the second part
        if padding_type == 'reflect':
            conv_fin += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_fin += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError(f'padding [{padding_type}] is not implemented')
        # Add a convolutional layer and a normalization layer for the second part
        conv_fin += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return conv_block, adaptive, conv_fin

    def forward(self, x, time_cond, z):
        """
        Forward function (with skip connections).

        Args:
            x (torch.Tensor): The input feature tensor.
            time_cond (torch.Tensor): The time conditioning vector.
            z (torch.Tensor): The style vector.

        Returns:
            torch.Tensor: The output feature tensor after passing through the block.
        """
        # Process the time conditioning vector
        time_input = self.Dense_time(time_cond)
        # Pass the input through the convolutional block
        for n, layer in enumerate(self.conv_block):
            out = layer(x)
            # Add the time input after the first layer of the convolutional block
            if n == 0:
                out += time_input[:, :, None, None]
        # Apply the adaptive layer
        out = self.adaptive(out, z)

        # Convert the feature map to a sequence for input to the Transformer
        b, c, h, w = out.shape
        out_seq = out.flatten(2).transpose(1, 2)  # [b, h*w, c]
        # Pass the sequence through the Transformer encoder layer
        out_seq = self.transformer_layer(out_seq)
        # Convert the sequence back to a feature map
        out = out_seq.transpose(1, 2).view(b, c, h, w)

        # Pass the output through the final convolutional layers
        for layer in self.conv_fin:
            out = layer(out)
        # Add skip connections
        out = x + out
        return out
        

class ResnetBlock_cond(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias,temb_dim,z_dim):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock_cond, self).__init__()
        self.conv_block,self.adaptive,self.conv_fin = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias,temb_dim,z_dim)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias,temb_dim,z_dim):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        
        self.conv_block = nn.ModuleList()
        self.conv_fin = nn.ModuleList()
        p = 0
        if padding_type == 'reflect':
            self.conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            self.conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        
        self.conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]
        self.adaptive = AdaptiveLayer(dim,z_dim) 
        self.conv_fin += [nn.ReLU(True)]
        if use_dropout:
            self.conv_fin += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            self.conv_fin += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            self.conv_fin += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        self.conv_fin += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]
        
        self.Dense_time = nn.Linear(temb_dim, dim)
        # self.Dense_time.weight.data = default_init()(self.Dense_time.weight.data.shape)
        nn.init.zeros_(self.Dense_time.bias)
        
        self.style = nn.Linear(z_dim, dim * 2)

        self.style.bias.data[:dim] = 1
        self.style.bias.data[dim:] = 0
        
        return self.conv_block,self.adaptive,self.conv_fin

    def forward(self, x,time_cond,z):
        
        time_input = self.Dense_time(time_cond)
        for n,layer in enumerate(self.conv_block):
            out = layer(x)
            if n==0:
                out += time_input[:, :, None, None]
        out = self.adaptive(out,z)
        for layer in self.conv_fin:
            out = layer(out)
        """Forward function (with skip connections)"""
        out = x + out  # add skip connections
        return out
###############################################################################
# Helper Functions
###############################################################################
def get_filter(filt_size=3):
    if(filt_size == 1):
        a = np.array([1., ])
    elif(filt_size == 2):
        a = np.array([1., 1.])
    elif(filt_size == 3):
        a = np.array([1., 2., 1.])
    elif(filt_size == 4):
        a = np.array([1., 3., 3., 1.])
    elif(filt_size == 5):
        a = np.array([1., 4., 6., 4., 1.])
    elif(filt_size == 6):
        a = np.array([1., 5., 10., 10., 5., 1.])
    elif(filt_size == 7):
        a = np.array([1., 6., 15., 20., 15., 6., 1.])

    filt = torch.Tensor(a[:, None] * a[None, :])
    filt = filt / torch.sum(filt)

    return filt


class Downsample(nn.Module):
    def __init__(self, channels, pad_type='reflect', filt_size=3, stride=2, pad_off=0):
        super(Downsample, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [int(1. * (filt_size - 1) / 2), int(np.ceil(1. * (filt_size - 1) / 2)), int(1. * (filt_size - 1) / 2), int(np.ceil(1. * (filt_size - 1) / 2))]
        self.pad_sizes = [pad_size + pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride - 1) / 2.)
        self.channels = channels

        filt = get_filter(filt_size=self.filt_size)
        self.register_buffer('filt', filt[None, None, :, :].repeat((self.channels, 1, 1, 1)))

        self.pad = get_pad_layer(pad_type)(self.pad_sizes)

    def forward(self, inp):
        if(self.filt_size == 1):
            if(self.pad_off == 0):
                return inp[:, :, ::self.stride, ::self.stride]
            else:
                return self.pad(inp)[:, :, ::self.stride, ::self.stride]
        else:
            return F.conv2d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])


class Upsample2(nn.Module):
    def __init__(self, scale_factor, mode='nearest'):
        super().__init__()
        self.factor = scale_factor
        self.mode = mode

    def forward(self, x):
        return torch.nn.functional.interpolate(x, scale_factor=self.factor, mode=self.mode)


class Upsample(nn.Module):
    def __init__(self, channels, pad_type='repl', filt_size=4, stride=2):
        super(Upsample, self).__init__()
        self.filt_size = filt_size
        self.filt_odd = np.mod(filt_size, 2) == 1
        self.pad_size = int((filt_size - 1) / 2)
        self.stride = stride
        self.off = int((self.stride - 1) / 2.)
        self.channels = channels

        filt = get_filter(filt_size=self.filt_size) * (stride**2)
        self.register_buffer('filt', filt[None, None, :, :].repeat((self.channels, 1, 1, 1)))

        self.pad = get_pad_layer(pad_type)([1, 1, 1, 1])

    def forward(self, inp):
        ret_val = F.conv_transpose2d(self.pad(inp), self.filt, stride=self.stride, padding=1 + self.pad_size, groups=inp.shape[1])[:, :, 1:, 1:]
        if(self.filt_odd):
            return ret_val
        else:
            return ret_val[:, :, :-1, :-1]


def get_pad_layer(pad_type):
    if(pad_type in ['refl', 'reflect']):
        PadLayer = nn.ReflectionPad2d
    elif(pad_type in ['repl', 'replicate']):
        PadLayer = nn.ReplicationPad2d
    elif(pad_type == 'zero'):
        PadLayer = nn.ZeroPad2d
    else:
        print('Pad type [%s] not recognized' % pad_type)
    return PadLayer


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x): return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def init_weights(net, init_type='normal', init_gain=0.02, debug=False):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if debug:
                print(classname)
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[], debug=False, initialize_weights=True):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
    if initialize_weights:
        init_weights(net, init_type, init_gain=init_gain, debug=debug)
    return net