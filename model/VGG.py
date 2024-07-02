# 导入必要的库
import json
import os

import torch
from torch import optim
from torch.optim import lr_scheduler
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import time
import math
from util_lib.partition import Partition

from torch.utils.checkpoint import checkpoint

class sub_vgg(nn.Module):
    def __init__(self,features,start_model,end_model,num_classes=1000,init_weights=True):
        super(sub_vgg, self).__init__()
        # print(features)
        self.features=features
        self.start_model=start_model
        self.end_model=end_model
        if end_model:
            self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )
        if init_weights:
            self._initialize_weights()

    def normal_for(self,x):
        y = self.features(x)
        return y
    
    def end_for(self,y):
        y = y.view(y.size(0), -1)
        y = self.classifier(y)
        return y

    def forward(self, x):
        
        y = checkpoint(self.normal_for,x)
        if self.end_model:
            y = y.view(y.size(0), -1)
            y = self.classifier(y)
        return y

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

in_channels = 3
def make_layers(cfg, batch_norm=True):
    global in_channels
    layers = [] 
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]     
            in_channels=v
    return nn.Sequential(*layers)

cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}
import copy
class Sub_VGG():
    def __init__(self,batchsize,layer_list,EPOCH,start_model,end_model,target_accumulation=None):
        self.cpu_model=sub_vgg(make_layers(layer_list),start_model,end_model)
        self.model=copy.deepcopy(self.cpu_model)
        self.accumulation=0 # Gpipe
        self.target_accumulation=target_accumulation # Gpipe
        self.layer_list=layer_list
        self.start_model=start_model
        self.end_model=end_model
        self.batchsize=batchsize
        self.EPOCH=EPOCH
        self.batch_x=None
        self.out_y=None
        self.optimizer=optim.Adam(self.model.parameters(), lr=0.0001)
        self.StepLR=optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.1)
        self.loss_func=nn.CrossEntropyLoss()

    def to(self,device):
        self.model.to(device)

    def forward(self,batch_x,batch_y,device):
        self.model.train()
        batch_x=torch.tensor(data=batch_x, dtype=torch.float, requires_grad=True, device=device)
        batch_y=Variable(batch_y).to(device)
        self.batch_x=batch_x
        if self.end_model:
            out_y=self.model.forward(self.batch_x)
            # print(out_y.shape)
            self.loss_func.to(device)
            out_y=self.loss_func(out_y,batch_y)
            # print(out_y.shape)
            self.out_y=out_y
        else:          
            out_y=self.model.forward(self.batch_x)
            self.out_y=out_y
        # print(self.out_y.shape)
        return self.out_y
    
    def backward(self,device=None,grad_y=None):
        self.out_y.to(device)
        if self.end_model:
            self.optimizer.zero_grad()
            self.out_y.backward()
        else:
            self.optimizer.zero_grad()
            grad_y=grad_y.to(device)
            self.out_y.backward(grad_y)

    def Micro_backward_step(self,grad_y=None,device=None):
        self.accumulation+=1
        if self.accumulation==1:
            self.optimizer.zero_grad()
            if self.end_model:                
                self.out_y.backward()
            else:
                grad_y=grad_y.to(device)
                self.out_y.backward(grad_y)
        elif self.accumulation==self.target_accumulation:
            if self.end_model:                
                self.out_y.backward()
            else:
                grad_y=grad_y.to(device)
                self.out_y.backward(grad_y)
            self.optimizer.step()
            self.accumulation=0
        else:
            if self.end_model:                
                self.out_y.backward()
            else:
                grad_y=grad_y.to(device)
                self.out_y.backward(grad_y)

    
    def step(self):
        self.optimizer.step()

    def swap(self,SubModel):
        SubModel.out_y=(self.out_y.to('cpu'))

    # def no_grad(self,device):
    #     self.model=torch.tensor(data=self.model., dtype=torch.float, requires_grad=False, device=device)
    def clone(self,device):
        return self.model.clone().to(device)

def Partition_Model_VGG(partition_lists,batch_size,EPOCH,Micro_number=None):
    Sub_Model_List=[]
    sub_model=None
    for idx,layer_list in enumerate(partition_lists):
        if idx==0:
            sub_model=Sub_VGG(batch_size,layer_list,EPOCH,True,False,target_accumulation=Micro_number)
        elif idx==len(partition_lists)-1:
            global in_channels
            in_channels=partition_lists[idx-1][-2]
            sub_model=Sub_VGG(batch_size,layer_list,EPOCH,False,True,target_accumulation=Micro_number)
        else:
            in_channels=partition_lists[idx-1][-2]
            sub_model=Sub_VGG(batch_size,layer_list,EPOCH,False,False,target_accumulation=Micro_number)
        Sub_Model_List.append(sub_model)
        in_channels=3
    return Sub_Model_List










