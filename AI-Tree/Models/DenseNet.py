import torch
from torch import nn
from torch.nn import functional as F

def conv_block(input_channels,num_channels):
    net = nn.Sequential(
        nn.BatchNorm2d(input_channels),
        nn.ReLU(),
        nn.Conv2d(input_channels,num_channels,kernel_size=3,padding=1)
    )
    return net

class DenseBlock(nn.Module):
    def __init__(self,num_convs,input_channels,num_channels):
        super(DenseBlock,self).__init__()
        layer = []
        for i in range(num_convs):
            layer.append(
                conv_block(num_channels*i+input_channels,num_channels)
            )
        self.net = nn.Sequential(*layer)
    
    def forward(self,x):
        for block in self.net:
            y = block(x)
            x = torch.cat((x,y),dim=1)
            # 将通道数直接拼接，ResNet算是叠加，DenseNet是拼接
        return x
    
def transition_block(input_ch,num_ch):
    net = nn.Sequential(
        nn.BatchNorm2d(input_ch),
        nn.ReLU(),
        nn.Conv2d(input_ch,num_ch,kernel_size=1),
        nn.AvgPool2d(kernel_size=2,stride=2)
    )
    # 目的就是先用1*1卷积减少特征通道数，然后再用池化减少图像宽高从而减少参数量
    return net

class Build(nn.Module):
    def __init__(self, num_classes = 4, init_weights = False):
        super(Build,self).__init__()
        b1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3), 
            nn.BatchNorm2d(64), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        # b1和ResNet几乎一模一样。

        num_ch,growth_rate = 64,32
        num_conv_in_dense_blocks = [4,4,4,4]
        blks =[]
        for i,num_convs in enumerate(num_conv_in_dense_blocks):
            blks.append(DenseBlock(num_convs,num_ch,growth_rate))
            # growth_rate其实就是单个稠密块的num_channels
            num_ch += num_convs * growth_rate
            if i != len(num_conv_in_dense_blocks)-1:
            # 也就是如果不是最后一个的话
                blks.append(transition_block(num_ch,num_ch//2))
                num_ch = num_ch // 2
                # // 代表整除
        self.net = nn.Sequential(
            b1,
            *blks,
            nn.BatchNorm2d(num_ch),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(num_ch,10)
        )
        if init_weights:
            self._initialize_weights()

    def forward(self,x):
        x = self.net(x)
        return x
    
    def _initialize_weights(self):
        for i in self.modules():
            if isinstance(i,nn.Conv2d):
                nn.init.xavier_uniform_(i.weight)
                if i.bias is not None:
                    nn.init.constant_(i.bias,0)
                elif isinstance(i,nn.Linear):
                    nn.init.xavier_uniform_(i.weight)
                    nn.init.constant_(i.bias,0)
    