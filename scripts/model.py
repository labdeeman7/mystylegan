### use the bilinear instead of the blur. 
# && not sure
# ?? ask
# ** notes. 
# normal comment that should make it into the repo.
# @@ need to Turn it to this to make it inline with official implementation.
import torch
from torch import nn
from torch.nn import functional as F

#** a form of normalization used in progan instead of batch normalization
#** normalizes the feature vectors in each pixel to unit length, applied after the convolutional layers, this is changed in style gan to adaptive instance normalization
#** only used at the begining of the mapping network for styleGAN to normalize the z-inout
class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input / torch.sqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)

#** check noise again. 
class InjectNoise(nn.Module):
    def __init__(self, n_channels, device="cpu"):
        super().__init__()

        # self.device = device

        self.weight = nn.Parameter(torch.randn(1, n_channels, 1, 1)).cuda()  # ** learned per feature scaling factors. notice, 1 value per channel. (B)
 
    def forward(self, image): # @@ need to change noise with changeable value, so as to show experiment.
        return image +  torch.randn(image.size(0), 1, image.size(2), image.size(3)).cuda() * self.weight 

class AdaIN(nn.Module):
    def __init__(self, w_dim, n_channels, device="cpu"):
        super(AdaIN, self).__init__()

        self.device = device

        # ** each feature map normalizzed separately and scaled and biased 
        #** using corresponding scalar components from style y.
        self.norm = nn.InstanceNorm2d(n_channels) #** this just does a normalization. no learnable parameters if affine parameter is not included. 
        # && it was used in the implementations I saw, but is this correct, it already prenormalizes it with the channels mean and std before we 
        # && perform the styled normalization in the 

        

        #learned affine transformation for the styles. 
        #each feature map is normalized. seaprately, so we learning n_channel affine transformations. 

        # @@ y_a_i and y_b_i, add equalized linear
        # self.style = EqualLinear(w_dim, n_channels * 2)
        # self.style.linear.bias.data[:in_channel] = 1 # for the mean
        # self.style.linear.bias.data[in_channel:] = 0 # for the standard dev. 
        # ?? what is the reason for making the bias one and zero??

        self.style_scale = nn.Linear(w_dim, n_channels)   
        self.style_shift = nn.Linear(w_dim, n_channels)

    def forward(self, image, w): #w is the style.
        # print("type of w is", type(w))
        # print("size of w is", w.size()) 
        # print(self.style_scale)
        scale = self.style_scale(w)[:, :, None, None] #** batch and channel
        shift = self.style_shift(w)[:, :, None, None] #** batch and channel 
        return (self.norm(image) * scale) + shift


# class ConstantInput(nn.Module):
#     def __init__(self, channel, size=4):
#         super().__init__()

#         self.input = nn.Parameter(torch.randn(1, channel, size, size))

#     #note that it is the input
#     def forward(self, input):
#         batch = input.shape[0]
#         out = self.input.repeat(batch, 1, 1, 1) # for the batch size. Use the same constant input

#         return out


# should be 8 fully connected layers. with 512 input mapping to 512 output
# @@ needs to be adapted for mean _style.

# remeber that we can have more than one style used. 
# For MNIST use a smaller value for the work, like n_mlp should be 5 or 6. and hidden_size = 32.  
class MappingNetwork(nn.Module):
    def __init__(self, z_dim, w_dim, hidden_size=512, n_mlp=8):  
        super().__init__()

        middle_layers = n_mlp - 2

        mapping_network = [PixelNorm()]
        mapping_network.append(nn.Linear(z_dim, hidden_size))
        mapping_network.append(nn.LeakyReLU(0.2))
        # mapping_network.append(EqualLinear(z_dim, hidden_size), nn.LeakyReLU(0.2)) # @@ equallinear
        
        for i in range(middle_layers):
            # mapping_network.append(EqualLinear(hidden_size, hidden_size))  # @@ equallinear
            mapping_network.append(nn.Linear(hidden_size, hidden_size))
            mapping_network.append(nn.LeakyReLU(0.2))

        mapping_network.append(
            nn.Linear(hidden_size, w_dim)
        )

        self.mapping_network = nn.Sequential(*mapping_network)

    def forward(self, input):
        #for either one or two styles
        return self.mapping_network(input)


# The styled conv block. 
# @@ at equallr for linear and conv block. 
# @@ Fused scaling was only used for greater than 128x128 in the original codebase. It was called auto, check lie 197 of netwoorks_stylegan.py 
class StyledConvBlock(nn.Module):
    def __init__(self, w_dim, in_channel, out_channel, kernel_size=3, padding=1, upsample=False, device="cpu"):
        super().__init__()

        # if initial:
        #     self.conv1 = ConstantInput(in_channel)

        if upsample:
            self.conv1 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear"), 
                nn.Conv2d(
                    in_channel, out_channel, kernel_size, padding=padding
                ),
            )
        else: #?? check why. 
            self.conv1 = nn.Conv2d(
                in_channel, out_channel, kernel_size, padding=padding
            )

        self.noise1 = InjectNoise(out_channel)
        self.adain1 = AdaIN(w_dim, out_channel)
        self.lrelu1 = nn.LeakyReLU(0.2)

        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size, padding=padding)
        self.noise2 = InjectNoise(out_channel)
        self.adain2 = AdaIN(w_dim, out_channel)
        self.lrelu2 = nn.LeakyReLU(0.2)

    def forward(self, input, style):
        out = self.conv1(input)
        out = self.noise1(out)
        out = self.lrelu1(out)
        out = self.adain1(out, style)

        out = self.conv2(out)
        out = self.noise2(out)
        out = self.lrelu2(out)
        out = self.adain2(out, style)

        return out

# generator 
# @@ still need to understand mixing
class StyleGANGenerator(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channel, z_dim, mapping_hidden_size, w_dim,
                  n_mlp=8, device="cpu"): #** use hidden channels = 256 for 

        super().__init__()

        self.constant_input= nn.Parameter(torch.randn(1, in_channels, 4, 4))
    
        self.mapping_network = MappingNetwork(z_dim, w_dim, mapping_hidden_size, n_mlp)

        self.progression = nn.ModuleList(
            [
                StyledConvBlock(w_dim, in_channels, 128, kernel_size=3, padding=1),  # 4 
                StyledConvBlock(w_dim, 128, 64, kernel_size=3, padding=1, upsample=True),  # 8
                StyledConvBlock(w_dim, 64, 32, kernel_size=3, padding=1, upsample=True),  # 16
                StyledConvBlock(w_dim, 32, 16, kernel_size=3, padding=1, upsample=True)  # 32
            ]
        )

        #define them differently, they are three different convolutions, used at 3 different instances. 
        self.to_rgb = nn.ModuleList(
            [
                nn.Conv2d(128, out_channel, 1), #4
                nn.Conv2d(64, out_channel, 1), #8
                nn.Conv2d(32, out_channel, 1), #16
                nn.Conv2d(16, out_channel, 1), #32
            ]
        )

    def forward(self, z, batch_size, step=0, alpha=-1,):

        out = self.constant_input.repeat(batch_size, 1, 1, 1)
        style = self.mapping_network(z)

        for i, (conv, to_rgb) in enumerate(zip(self.progression, self.to_rgb)):

            if i > 0 and step > 0:
                out_prev = out

            out = conv(out, style)

            if i == step: #** for the latest step, 
                out = to_rgb(out) # makes sense. 

                if i > 0 and 0 <= alpha < 1:
                    skip_rgb = self.to_rgb[i - 1](out_prev) # the skip rgb is the one that is not complex, the one with the upsampled version, anf that is it.
                    skip_rgb = F.interpolate(skip_rgb, scale_factor=2, mode='nearest') #skip rgb, the direct version without convs just upsampled.
                    out = (1 - alpha) * skip_rgb + alpha * out # nice.

                break # break allows you to do 1, 1-2, 1-2-3 etc. 
            

        return out

class ConvBlockDownSample(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        padding = 1,
        kernel_size2=None,
        padding2=None,
        downsample=False,
    ):
        super().__init__()

        pad1 = padding
        pad2 = padding
        if padding2 is not None:
            pad2 = padding2

        kernel1 = kernel_size
        kernel2 = kernel_size
        if kernel_size2 is not None:
            kernel2 = kernel_size2

        #first conv
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel1, padding=pad1),
            nn.LeakyReLU(0.2),
        )

        # note that the convolutions are the same size for the fused out sampling. Also note that the blur is done after the nearest neighbor has been done. . 
        if downsample:
            # print("downsample true")
            self.conv2 = nn.Sequential(
                nn.Conv2d(out_channel, out_channel, kernel2, padding=pad2),
                nn.AvgPool2d(2), # downsampling been done here. 
                nn.LeakyReLU(0.2),
            )

        else:
            # print("downsample false")
            self.conv2 = nn.Sequential(
                nn.Conv2d(out_channel, out_channel, kernel2, padding=pad2),
                nn.LeakyReLU(0.2),
            )

    # foward pass of the normal conv
    def forward(self, input):
        # print("input size ", input.size())
        out = self.conv1(input)
        # print("after conv1 ", out.size())
        out = self.conv2(out)

        return out

class StyleGANDiscriminator(nn.Module):
    def __init__(self, in_size = 1, from_rgb_activate=False): #black and white or coloured.
        super().__init__()

        self.progression = nn.ModuleList(
            [
                ConvBlockDownSample(16, 32, 3, 1, downsample=True),  # 32
                ConvBlockDownSample(32, 64, 3, 1, downsample=True),  # 16
                ConvBlockDownSample(64, 128, 3, 1, downsample=True),  # 8
                ConvBlockDownSample(129, 128, 3, 1, 4, 0),  # 4
            ]
        )

        def make_from_rgb(out_channel):
            if from_rgb_activate:
                return nn.Sequential(nn.Conv2d(in_size, out_channel, 1), nn.LeakyReLU(0.2))

            else:
                return nn.Conv2d(in_size, out_channel, 1)

        self.from_rgb = nn.ModuleList(
            [
                make_from_rgb(16),
                make_from_rgb(32),
                make_from_rgb(64),
                make_from_rgb(128)
            ]
        )

        self.n_layer = len(self.progression)

        self.linear = nn.Linear(128, 1)

    def forward(self, input, step=0, alpha=-1):
        for i in range(step, -1, -1):
            index = self.n_layer - i - 1

            # print(step)
            # print(i)
            if i == step:
                out = self.from_rgb[index](input)

            if i == 0: #** why is this done?
                out_std = torch.sqrt(out.var(0, unbiased=False) + 1e-8)
                mean_std = out_std.mean()
                mean_std = mean_std.expand(out.size(0), 1, 4, 4)
                out = torch.cat([out, mean_std], 1) #** The mean of the standard deviation is added as an activation layer, why?

            out = self.progression[index](out)

            if i > 0:
                if i == step and 0 <= alpha < 1:
                    skip_rgb = F.avg_pool2d(input, 2)
                    skip_rgb = self.from_rgb[index + 1](skip_rgb)

                    out = (1 - alpha) * skip_rgb + alpha * out

        out = out.squeeze(2).squeeze(2)
        # print("last step, out size", out.size())
        out = self.linear(out)

        return out