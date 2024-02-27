from torch import nn
from torch.nn import functional as F

class Residual(nn.Module):
    def __init__(self,input_channels,output_channels,strides=1):
        # conv1x1代表用不用1*1卷积核来降低参数复杂度
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels,output_channels,kernel_size=3,padding=1,stride=strides)
        # 计算复杂度和参数总数有关，参数量 = 卷积核高 * 卷积核宽 * 输入通道数 * 输出通道数
        self.conv2 = nn.Conv2d(output_channels,output_channels,kernel_size=3,padding=1)
        if input_channels != output_channels:
            self.conv3 = nn.Conv2d(input_channels,output_channels,kernel_size=1,stride=strides)
        else:
            self.conv3 = None
        # 1*1卷积的目的单纯是让通道数一致，方便后续相加
        self.bn1 = nn.BatchNorm2d(output_channels)
        # 使得输入的图像批量标准化，方便收敛？其中有几个参数需要学习
        self.bn2 = nn.BatchNorm2d(output_channels)
    def forward(self,x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        if self.conv3:
            x = self.conv3(x)
        y += x
        return F.relu(y)    

def resnet_block(input_channels, num_channels, num_residuals):
    blk = [] 
    for i in range(num_residuals): 
        if i == 0 and input_channels != num_channels: 
            blk.append(Residual(input_channels, num_channels, strides=2))
        else: 
            blk.append(Residual(num_channels, num_channels))
    return blk

class Build(nn.Module):
    def __init__(self, num_classes = 4, init_weights = False):
        super(Build,self).__init__()
        b1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3), 
            nn.BatchNorm2d(64), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        b2 = nn.Sequential(*resnet_block(64, 64, 2)) 
        b3 = nn.Sequential(*resnet_block(64, 128, 2)) 
        b4 = nn.Sequential(*resnet_block(128, 256, 2))
        b5 = nn.Sequential(*resnet_block(256, 512, 2))
        self.net = nn.Sequential(
            b1, b2, b3, b4, b5,
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(), 
            nn.Linear(512, num_classes)
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