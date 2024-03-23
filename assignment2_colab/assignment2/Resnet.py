import torch
import torch.nn as nn

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, identity_downsample = None, stride=1):
        super(Block, self).__init__()
        self.expansion = 4
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        
        self.relu = nn.ReLU()
        
        self.identity_downsample = identity_downsample
        
    def forward(self, x):
        identity = x
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        
        if self.identity_downsample:
            identity = self.identity_downsample(identity) 
                       
        x += identity
        x = self.relu(x)
        return x


class ResNet(nn.Module):
    def __init__(self, Block, layers, img_channel, num_classes):
        super(ResNet, self).__init__()
        #define previous resnet layer
        self.conv1 = nn.Conv2d(in_channels=img_channel, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)        
        
        self.in_channels = 64    # in channel of resnet layer
        #resnmet layer 
        self.layer1 = self.make_layer(Block, layers[0], out_channels=64, stride = 1)
        self.layer2 = self.make_layer(Block, layers[1], out_channels=128, stride = 2)
        self.layer3 = self.make_layer(Block, layers[2], out_channels=256, stride = 2)
        self.layer4 = self.make_layer(Block, layers[3], out_channels=512, stride = 2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(2048, num_classes)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        
        return x
    
    
    def make_layer(self, Block, num_res_block, out_channels, stride):
        layer = []
        identity_downsample = None
        if stride != 1 or self.in_channels != out_channels * 4:
            identity_downsample = nn.Sequential(
                                        nn.Conv2d(self.in_channels, out_channels * 4, kernel_size=1, stride = stride),
                                        nn.BatchNorm2d(out_channels * 4)
                                    )
        
        layer.append(Block(self.in_channels, out_channels,identity_downsample = identity_downsample, stride = stride))
        
        self.in_channels = out_channels*4
        
        for i in range (num_res_block-1):
            block = Block(self.in_channels, out_channels)
            layer.append(block)
        
        return nn.Sequential(*layer)
    

def ResNet18(img_channel = 3, num_classes = 100):
    return ResNet(Block=Block, layers = [2,2,2,2], img_channel=img_channel, num_classes=num_classes)