from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math
import numpy as np

def convbn(in_planes, out_planes, kernel_size, stride, pad, dilation):
    """Constructs a convolutional layer followed by batch normalization.

    Args:
        in_planes (int): Number of input channels.
        out_planes (int): Number of output channels.
        kernel_size (int or tuple): Size of the convolutional kernel.
        stride (int or tuple): Stride of the convolution.
        pad (int or tuple): Padding added to the input.
        dilation (int or tuple): Dilation rate of the convolution.

    Returns:
        nn.Sequential: A sequential module consisting of a convolutional layer followed by batch normalization.
    """
    # 批量归一化的主要思想是对每一层的输入进行归一化处理，使得结果的均值为0，方差为1。
    # 这样可以防止网络中的数值不稳定，例如梯度消失或梯度爆炸等问题。  
    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=dilation if dilation > 1 else pad, dilation = dilation, bias=False),
                         nn.BatchNorm2d(out_planes))


def convbn_3d(in_planes, out_planes, kernel_size, stride, pad):
    """Constructs a 3D convolutional layer followed by batch normalization.

    Args:
        in_planes (int): Number of input channels.
        out_planes (int): Number of output channels.
        kernel_size (int or tuple): Size of the convolutional kernel.
        stride (int or tuple): Stride of the convolution.
        pad (int or tuple): Padding added to the input.

    Returns:
        nn.Sequential: A sequential module consisting of a 3D convolutional layer followed by batch normalization.
    """    
    return nn.Sequential(nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, padding=pad, stride=stride,bias=False),
                         nn.BatchNorm3d(out_planes))

class BasicBlock(nn.Module):
    """
    BasicBlock is a basic building block for the PSMNet model.

    Args:
        inplanes (int): Number of input channels.
        planes (int): Number of output channels.
        stride (int): Stride value for the convolutional layers.
        downsample (nn.Module, optional): Downsample layer. Default is None.
        pad (int): Padding value for the convolutional layers.
        dilation (int): Dilation value for the convolutional layers.

    Attributes:
        expansion (int): Expansion factor for the number of output channels.

    """

    expansion = 1

    def __init__(self, inplanes, planes, stride, downsample, pad, dilation):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Sequential(convbn(inplanes, planes, 3, stride, pad, dilation),
                                   nn.ReLU(inplace=True))

        self.conv2 = convbn(planes, planes, 3, 1, pad, dilation)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x

        return out

class disparityregression(nn.Module):
    """
    Disparity Regression module.
    
    Args:
        maxdisp (int): Maximum disparity value.
    """
    
    def __init__(self, maxdisp):
        super(disparityregression, self).__init__()
        self.disp = torch.Tensor(np.reshape(np.array(range(maxdisp)),[1, maxdisp,1,1])).cuda()

    def forward(self, x):
        """
        Forward pass of the disparity regression module.
        
        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            torch.Tensor: Output tensor.
        """
        out = torch.sum(x*self.disp.data,1, keepdim=True)
        return out

class feature_extraction(nn.Module):
    def __init__(self):
        """
        Initializes the feature_extraction module.

        This module is responsible for extracting features from input images using a series of convolutional layers.
        It consists of several layers including the first convolutional layer, four layers of BasicBlocks, and four branches.

        Args:
            None

        Returns:
            None
        """
        super(feature_extraction, self).__init__()
        self.inplanes = 32
        """
        def convbn(in_planes, out_planes, kernel_size, stride, pad, dilation)
        Args:
            in_planes (int): Number of input channels.
            out_planes (int): Number of output channels.
            kernel_size (int or tuple): Size of the convolutional kernel.
            stride (int or tuple): Stride of the convolution.
            pad (int or tuple): Padding added to the input.
            dilation (int or tuple): Dilation rate of the convolution.
        """
        # CNN in Table 1
        
        # Downsampling is performed by conv0 1 and conv2 1 with stride of 2.
        
        # conv0_1 is convbn(3, 32, 3, 2, 1, 1)
        # conv0_2 is convbn(32, 32, 3, 1, 1, 1)
        # conv0_3 is convbn(32, 32, 3, 1, 1, 1)
        self.firstconv = nn.Sequential(convbn(3, 32, 3, 2, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True))
        
        """
        def _make_layer(self, block, planes, blocks, stride, pad, dilation)
        Args:
            block (nn.Module): The type of block to be used in the layer.
            planes (int): The number of output channels for each block.
            blocks (int): The number of blocks in the layer.
            stride (int): The stride value for the first block.
            pad (int): The padding value for the first block.
            dilation (int): The dilation value for the first block.扩张值
        """
        # conv1_x, output channels=32, blocks number=3
        self.layer1 = self._make_layer(BasicBlock, 32, 3, 1,1,1)
        # conv2_x, Downsampling is performed by conv0 1 and conv2 1 with stride of 2.
        self.layer2 = self._make_layer(BasicBlock, 64, 16, 2,1,1)
        # conv3_x 
        # * denote that we use half the dilated rate of dilated convolution
        self.layer3 = self._make_layer(BasicBlock, 128, 3, 1,1,1)
        # conv4_x, output channels=128, blocks number=3, dilation=2
        self.layer4 = self._make_layer(BasicBlock, 128, 3, 1,1,2)

        # SPP module in Table 1
        # 将每个64x64的区域减少到一个单独的像素，然后通过一个1x1的卷积层将通道数从128减少到32
        # 这里下采样，后面上采样
        # 分辨率降为1/64
        self.branch1 = nn.Sequential(nn.AvgPool2d((64, 64), stride=(64,64)),
                                     convbn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.branch2 = nn.Sequential(nn.AvgPool2d((32, 32), stride=(32,32)),
                                     convbn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.branch3 = nn.Sequential(nn.AvgPool2d((16, 16), stride=(16,16)),
                                     convbn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.branch4 = nn.Sequential(nn.AvgPool2d((8, 8), stride=(8,8)),
                                     convbn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.lastconv = nn.Sequential(convbn(320, 128, 3, 1, 1, 1),
                                      nn.ReLU(inplace=True),
                                      # in_channels=128和out_channels=32
                                      # 卷积层将会有32个卷积核，每个卷积核的深度为128
                                      # weight 是 128x32
                                      # 
                                      nn.Conv2d(128, 32, kernel_size=1, padding=0, stride = 1, bias=False))

    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        """
        Creates a layer consisting of multiple blocks.

        Args:
            block (nn.Module): The type of block to be used in the layer.
            planes (int): The number of output channels for each block.
            blocks (int): The number of blocks in the layer.
            stride (int): The stride value for the first block.
            pad (int): The padding value for the first block.
            dilation (int): The dilation value for the first block.

        Returns:
            nn.Sequential: The layer consisting of multiple blocks.
        """
        # 如果步长不为1，或者输入通道数与输出通道数不匹配，那么就需要进行下采样
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
           downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),)
        # 创建一个空的列表layers
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
        # BasicBlock.expansion=1
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,1,None,pad,dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Performs forward pass of the feature_extraction module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after feature extraction.
        """
        # CNN in Table 1
        output      = self.firstconv(x)
        output      = self.layer1(output)
        output_raw  = self.layer2(output)
        output      = self.layer3(output_raw)
        output_skip = self.layer4(output)
        # SPP module in Table 1
        # incorporate hierarchical context information
        # 为了提取不同尺度的特征，我们将输入图像分别缩放到1/2、1/4、1/8和1/16的分辨率
        # we design four fixed-size average pooling blocks for SPP
        # downsample
        output_branch1 = self.branch1(output_skip)
        # upsample
        output_branch1 = F.upsample(output_branch1, (output_skip.size()[2],output_skip.size()[3]),mode='bilinear')

        output_branch2 = self.branch2(output_skip)
        output_branch2 = F.upsample(output_branch2, (output_skip.size()[2],output_skip.size()[3]),mode='bilinear')

        output_branch3 = self.branch3(output_skip)
        output_branch3 = F.upsample(output_branch3, (output_skip.size()[2],output_skip.size()[3]),mode='bilinear')

        output_branch4 = self.branch4(output_skip)
        output_branch4 = F.upsample(output_branch4, (output_skip.size()[2],output_skip.size()[3]),mode='bilinear')
        # concatenate the output of the SPP module
        output_feature = torch.cat((output_raw, output_skip, output_branch4, output_branch3, output_branch2, output_branch1), 1)
        # fusion: 3x1,128 1x1,32 convolution
        output_feature = self.lastconv(output_feature)

        return output_feature



