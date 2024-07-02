from __future__ import print_function, division
import torch.nn.functional as F
import torch
import copy
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torchvision import models
from torch.autograd import Variable
import os
import csv
import codecs
import numpy as np
import time
from torch.utils.data import random_split
from torch.utils.data import DataLoader

import argparse

classes_num=1000

class BasicBlock(nn.Module):
    '''这个函数适用于构成ResNet18和ResNet34的block'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out

class Bottleneck(nn.Module):
    '''这个函数适用于构成ResNet50及更深层模型的的block'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = torch.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out

class resnet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=classes_num):
        super(resnet, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.classifier = nn.Linear(512 * block.expansion, num_classes)


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out_x1 = self.relu(out)
        out_x2 = self.maxpool(out_x1)
        out1 = self.layer1(out_x2)  # 56*56      4
        out2 = self.layer2(out1)  # 28*28        4
        out3 = self.layer3(out2)  # 14*14        4
        out4 = self.layer4(out3)  # (512*7*7)     4
        # out5 = F.avg_pool2d(out4, 4)#平均池化
        out5 = self.avgpool(out4)
        out6 = out5.view(out5.size(0), -1)  # view()函数相当于numpy中的reshape
        out7 = self.classifier(out6)  # 平均池化全连接分类
        return out7

class ResNet():
    def __init__(self,modeltype,batchsize,Load_flag,model_dir,epoch_dir,EPOCH):
        self.modeltype=modeltype 
        self.model=None
        self.batchsize=batchsize
        if modeltype==18:
            self.model=resnet(BasicBlock, [2, 2, 2, 2])
        if modeltype==34:
            self.model=resnet(BasicBlock, [3, 4, 6, 3])
        if modeltype==50:
            self.model=resnet(Bottleneck, [3, 4, 6, 3])
        if modeltype==101:
            self.model=resnet(Bottleneck, [3, 4, 23, 3])
        if modeltype==152:
            self.model=resnet(Bottleneck, [3, 8, 36, 3])
        self.optimizer=optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
        self.StepLR=optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.1)
        self.loss_func=nn.CrossEntropyLoss()
        self.model_dir=model_dir
        self.epoch_dir=epoch_dir
        self.finish_epoch=0
        self.EPOCH=EPOCH

        if Load_flag==True:
            self.model.load_state_dict(torch.load(self.model_dir)) 
            self.finish_epoch=torch.load(epoch_dir)
            self.EPOCH=self.EPOCH-self.finish_epoch


    def train_iteration(self,batch_x,batch_y,device):
        self.model.train()
        train_loss = 0.
        train_acc = 0.
        batch_x = Variable(batch_x).to(device)
        batch_y = Variable(batch_y).to(device)
        self.optimizer.zero_grad()
        out = self.model(batch_x).to(device)
        loss = self.loss_func(out, batch_y)
        loss.backward()
        train_loss += loss.item()
        pred = torch.max(out, 1)[1]
        train_correct = (pred == batch_y).sum()
        train_acc += train_correct.item()

    def to(self,device):
        self.model.to(device)  

    def get_gradients(self):
        grad_list=[]
        for p in self.model.parameters():
            grad=p.grad
            grad_list.append(grad)
        return grad_list
    
    def zero_grad(self):
        self.optimizer.zero_grad()
    
    def load_gradients(self,grads):
        for tmp_g,tmp_p in zip(grads, self.model.named_parameters()):
                if tmp_g is not None:
                    tmp_p[1].grad = tmp_g

    def step(self):
        self.optimizer.step()

    def load_dict(self,weight):
        self.model.load_state_dict(weight)

    def get_dict(self):
        weight=self.model.state_dict()
        return weight
    
    def save_model(self):
        torch.save(self.model.state_dict(), self.model_dir)

    def save_epoch(self,epoch):
        torch.save(epoch+self.finish_epoch, self.epoch_dir, _use_new_zipfile_serialization=False)