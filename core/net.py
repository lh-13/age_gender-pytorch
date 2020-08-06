'''
网络层：用于年龄和性别的预测
author:lh-13
date:2020.0802
'''


import torch 
import torch.nn as nn 
import torch.nn.functional as F 


class AgeGenderNet(nn.Module):
    def __init__(self):
        super(AgeGenderNet, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            
            nn.Conv2d(32, 64, 3, stride=1, padding=1), 
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0), 
            
            nn.Conv2d(64, 96, 3, stride=1, padding=1), 
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

            nn.Conv2d(96, 128, 3, stride=1, padding=1), 
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

            nn.Conv2d(128, 196, 3, stride=1, padding=1), 
            nn.BatchNorm2d(196),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        )
        #输出size为1*1
        self.global_max_pooling = nn.AdaptiveMaxPool2d(output_size=(1,1))

        self.age_fc_layers = nn.Sequential(
            nn.Linear(in_features=196, out_features=25, bias=True), #why out_features is 25
            nn.ReLU(),
            nn.Linear(in_features=25, out_features=1, bias=True),
            nn.Sigmoid()
        )
        self.gender_fc_layers = nn.Sequential(
            nn.Linear(in_features=196, out_features=25, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=25, out_features=2, bias=True),
        )


    def forward(self, x):
        x = self.conv_layer(x)
        x = self.global_max_pooling(x)
        x = x.view(-1, 196)
        age = self.age_fc_layers(x)
        gender = self.gender_fc_layers(x)

        return age, gender 
