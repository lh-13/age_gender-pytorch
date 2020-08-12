'''
resnet.py  implementation of the resnet using pytorch  
author:lh-13 
date:2020.0810
'''
import torch 
import torch.nn as nn  
import torch.nn.functional as F  

class BasicBlock(nn.Module):
    expansion = 1     
    def __init__(self,in_channle, out_channle, stride=1, downsample=None):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channle, out_channle, 3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channle)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(out_channle, out_channle, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channle)
        self.stride = stride 
        self.downsample = downsample

    def forward(self, x):
        residual = x  
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out     

class Resnet(nn.Module):
    def __init__(self, block, layers, num_classes, grayflag=False):
        super(Resnet, self).__init__()
        self.in_channel = 64   #初始时输入通道数量，(在_make_layer中要更新此值)
        if grayflag:
            in_dim = 1 
        else: 
            in_dim = 3 
        self.conv1 = nn.Conv2d(in_dim, 64, kernel_size=7, stride=2, padding=3)    
        #should be padding=3   why?-because x_输出=(x_输入+2*padding-Kernel_size)/stride,向上取整 即（224+2*3-7）/2 = 112
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()

        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)    #should be padding=1  ,同理如上，padding一定是1，否则无法使用输出为56
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)    #out_channel = 64,        
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)   #out_channel = 128（第一次repeat输入channel为上一次的out_channle即64）
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)   #同上
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)   

        #self.avgpool = nn.AvgPool2d(7, stride=1)     #kernel_size=7   why  因为到这里的时候输入size为[b, 7*7]
        #self.fc = nn.Linear(512*block.expansion, num_classes)
        self.avgpool = nn.AvgPool2d(2, stride=1) 
        self.age_fc_layers = nn.Sequential(
            nn.Linear(in_features=512, out_features=25, bias=True), #why out_features is 25
            nn.ReLU(),
            nn.Linear(in_features=25, out_features=1, bias=True),
            nn.Sigmoid()
        )
        self.gender_fc_layers = nn.Sequential(
            nn.Linear(in_features=512, out_features=25, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=25, out_features=2, bias=True),
        )


    def _make_layer(self, block, out_channel, repeat, stride=1):
        downsample = None
        #先判断是否需要下采样
        if (stride != 1 or self.in_channel != out_channel*block.expansion):
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, out_channel*block.expansion, kernel_size=1, stride=stride), 
                nn.BatchNorm2d(out_channel*block.expansion)
            )
        
        layers = []
        layers.append(block(self.in_channel, out_channel,stride, downsample))

        self.in_channel = out_channel*block.expansion

        for i in range(1, repeat):
            layers.append(block(self.in_channel, out_channel))     #这里的downsapmle使用默认值，也即None

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)   #如果所使用的数据集到这里已经是1*1,则不需要此avgpool(x)

        x = x.view(x.size(0), -1)
        age = self.age_fc_layers(x)
        gender = self.gender_fc_layers(x)

        return age, gender 

        # logits = self.fc(x)

        # prob = F.softmax(logits, dim=1)
        # return logits, prob


def resnet18(num_class=1):
    model = Resnet(BasicBlock, layers=[2, 2, 2, 2], num_classes=num_class, grayflag=0)
    return model 

def resnet34(num_class):
    model = Resnet(BasicBlock, layers=[3, 4, 6, 3], num_classes=num_class, grayflag=0)
    return model  


if __name__ == "__main__":
    x = torch.rand(4, 3, 224, 224)
    model = resnet18(10)
    out = model(x)

