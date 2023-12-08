from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math
from .submodule import *

class PSMNet(nn.Module):
    def __init__(self, maxdisp):
        super(PSMNet, self).__init__()
        self.maxdisp = maxdisp
        self.feature_extraction = feature_extraction()

########
        # des1,2,3,4 为残差模块，完全相同
        self.dres0 = nn.Sequential(convbn_3d(64, 32, 3, 1, 1),
                                     nn.ReLU(inplace=True),
                                     convbn_3d(32, 32, 3, 1, 1),
                                     nn.ReLU(inplace=True))

        self.dres1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1)) 

        self.dres2 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1))
 
        self.dres3 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1)) 

        self.dres4 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1)) 
 
        self.classify = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1,bias=False))

        # 初始化模型的权重。对于不同类型的层（如nn.Conv2d，nn.Conv3d，nn.BatchNorm2d，nn.BatchNorm3d和nn.Linear），使用不同的初始化策略。
        # 对于nn.Conv2d和nn.Conv3d卷积层，权重是根据卷积核的大小和输出通道的数量计算得到的。这里使用了一种称为"He initialization"的方法，
        # 这种方法特别适合ReLU激活函数。这种初始化方法的目的是保持每一层输出的方差一致，避免深层网络中的梯度消失或爆炸问题。
        # 对于nn.BatchNorm2d和nn.BatchNorm3d批量归一化层，权重被初始化为1，偏置被初始化为0。这是因为批量归一化层的目的是将输入数据标准化
        # （即，使得输入数据的均值为0，方差为1），初始化权重为1和偏置为0可以使得在训练初期，批量归一化层对输入数据的影响最小。
        # 对于nn.Linear全连接层，偏置被初始化为0。权重的初始化方法并未在这段代码中给出，可能在其他地方进行了初始化。
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1]*m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()


    def forward(self, left, right):
        # feature extraction
        refimg_fea    = self.feature_extraction(left)
        targetimg_fea = self.feature_extraction(right)

        # create cost volume
        cost = Variable(torch.FloatTensor(refimg_fea.size()[0], refimg_fea.size()[1]*2, self.maxdisp/4, refimg_fea.size()[2], refimg_fea.size()[3]).zero_(), volatile=not self.training).cuda()

        for i in range(self.maxdisp/4):
            if i > 0:
                cost[:, :refimg_fea.size()[1], i, :, i:] = refimg_fea[:, :, :, i:]
                cost[:, refimg_fea.size()[1]:, i, :, i:] = targetimg_fea[:, :, :, :-i]
            else:
                cost[:, :refimg_fea.size()[1], i, :, :]  = refimg_fea
                cost[:, refimg_fea.size()[1]:, i, :, :]  = targetimg_fea
        cost = cost.contiguous()
        
        # 3D CNN: basic module
        # figue1 中的橙色箭头代表 + cost0
        # 残差连接 之前的特征图与当前的特征图相加，
        # 可以使得网络更容易学习到残差部分，从而提高网络的性能和训练效果
        # 帮助网络更好地学习到特征图中的细节信息，从而提高视差估计的准确性
        cost0 = self.dres0(cost)
        cost0 = self.dres1(cost0) + cost0
        cost0 = self.dres2(cost0) + cost0
        cost0 = self.dres3(cost0) + cost0
        cost0 = self.dres4(cost0) + cost0

        cost = self.classify(cost0)
        cost = F.upsample(cost, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear')
        cost = torch.squeeze(cost, 1)
        pred = F.softmax(cost)
        
        pred = disparityregression(self.maxdisp)(pred)

        return pred
