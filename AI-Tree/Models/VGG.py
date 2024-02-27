import torch
from torch import nn

class Build(nn.Module):
    def __init__(self,num_classes=4,init_weights=False,version='vgg16'):
        super(Build,self).__init__()
        self.cfgs ={
            'vgg16':[64,64,'M',128,128,'M',256,256,256,'M',512,512,512,'M',512,512,512,'M']
        }
        layers = []
        input_channels = 3
        for i in self.cfgs[version]:
            if i == 'M':
                layers += [nn.MaxPool2d(kernel_size = 2,stride = 2)]
            else:
                conv2d = nn.Conv2d(input_channels,i,kernel_size = 3,padding = 1)
                layers += [conv2d, nn.ReLU(True)]
                input_channels = i
        self.BackboneNet = nn.Sequential(*layers)

        self.classifier = nn.Sequential(
            nn.Linear(512*7*7,4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096,4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096,num_classes)
        )

        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self):
        for i in self.modules():
            if isinstance(i,nn.Conv2d):
                nn.init.xavier_uniform_(i.weight)
                if i.bias is not None:
                    nn.init.constant_(i.bias,0)
                elif isinstance(i,nn.Linear):
                    nn.init.xavier_uniform_(i.weight)
                    nn.init.constant_(i.bias,0)
    
    def forward(self,x):
        x = self.BackboneNet(x)
        x = torch.flatten(x,start_dim=1)
        x = self.classifier(x)
        return x