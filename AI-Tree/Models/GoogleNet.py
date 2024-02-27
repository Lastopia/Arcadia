import torch
from torch import nn
from torch.nn import functional as F

class Inception(nn.Module):
    # 总共由四条输出通道，c1-c4分别是四条通道的目标输出通道数
    # 每一条输出通道都代表的一张卷积过后的特征图，所以输出通道数越多，表征能力越强，但计算量越大
    def __init__(self, in_channels,c1,c2,c3,c4,**kwargs):
        super(Inception,self).__init__(**kwargs)
        self.p1_1 = nn.Conv2d(in_channels,c1,kernel_size=1)
        # 第一条，仅1*1卷积层
        self.p2_1 = nn.Conv2d(in_channels,c2[0],kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0],c2[1],kernel_size=3,padding=1)
        # 第二条，1*1卷积层接3*3卷积层
        self.p3_1 = nn.Conv2d(in_channels,c3[0],kernel_size =1)
        self.p3_2 = nn.Conv2d(c3[0],c3[1],kernel_size=5,padding=2)
        # 第三条，1*1卷积层接5*5卷积层
        self.p4_1 = nn.MaxPool2d(kernel_size=3,stride=1,padding=1)
        self.p4_2 = nn.Conv2d(in_channels,c4,kernel_size=1)
        # 第四条，3*3最大池化，再加上1*1卷积

    # forward(data)函数是类的固有函数
    # Inception(data)其实就是等价于Inception.forward(data)
    def forward(self,x):
        # 激活函数的相关设定
        p1 = F.relu(self.p1_1(x))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        p4 = F.relu(self.p4_2(self.p4_1(x)))
        return torch.cat((p1,p2,p3,p4),dim=1)
        # 将经过四种不同卷积的结果通道进行叠加
# RestNet中的特殊模块
class Build(nn.Module):
    def __init__(self, num_classes = 4, init_weights = False):
        super(Build,self).__init__()

        b1 = nn.Sequential(
            nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3),
            # 第一个一般都是3，64
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        )

        b2 = nn.Sequential(
            nn.Conv2d(64,64,kernel_size=1),
            # 输入64通道，输出64通道，因为用的是1*1卷积核
            nn.ReLU(),
            nn.Conv2d(64,192,kernel_size=3,padding=1),
            # 输入64，输出192，因为用的是3*3卷积核
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        )

        b3 = nn.Sequential(
            Inception(192, 64, (96, 128), (16, 32), 32), 
            Inception(256, 128, (128, 192), (32, 96), 64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        b4 = nn.Sequential(
            Inception(480, 192, (96, 208), (16, 48), 64),
            Inception(512, 160, (112, 224), (24, 64), 64), 
            Inception(512, 128, (128, 256), (24, 64), 64), 
            Inception(512, 112, (144, 288), (32, 64), 64), 
            Inception(528, 256, (160, 320), (32, 128), 128),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        b5 = nn.Sequential(
            Inception(832, 256, (160, 320), (32, 128), 128), 
            Inception(832, 384, (192, 384), (48, 128), 128), 
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten()
        )

        self.net = nn.Sequential(b1, b2, b3, b4, b5, nn.Linear(1024, num_classes))
        # 模型已经编写好了

        if init_weights:
            self._initialize_weights()
        #是否对网络参数初始化

    # 前向传播
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
