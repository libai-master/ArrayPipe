# from __future__ import print_function, division
# import torch.nn.functional as F
# import torch
# import copy
# import torch.nn as nn
# import torch.optim as optim
# import torchvision


# import numpy as np



# decive0=torch.device("cuda:0")
# decive1=torch.device("cuda:1")
# tensor0 = torch.zeros([2,3],dtype=float).to(decive0)
# tensor1 = torch.ones([2,2],dtype=float).to(decive1)

# print(tensor0)
# print(tensor1)
# tensor10=tensor1.cuda(0)
# tenser_list=[tensor0,tensor10]
# tensor_list=torch.stack(tenser_list)
# # list = tensor_list.cpu().numpy().tolist()

# # for i in tensor_list:
# print(tensor_list)

import torch 
import torch.nn as nn
import numpy as np
import torch.optim as optim

import os
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm


# 设置一下数据集   数据集的构成是随机两个整数，形成一个加法的效果 input1 + input2 = label
class TrainDataset(Dataset):
    def __init__(self):
        super(TrainDataset, self).__init__()
        self.data = []
        for i in range(1,1000):
            for j in range(1,1000):
                self.data.append([i,j])
    def __getitem__(self, index):
        input_data = self.data[index]
        label = input_data[0] + input_data[1]
        return torch.Tensor(input_data),torch.Tensor([label])
    def __len__(self):
        return len(self.data)

class TestNet(nn.Module):
    def __init__(self):
        super(TestNet, self).__init__()
        self.net1 = nn.Linear(2,1)

    def forward(self, x):
        x = self.net1(x)
        return x

def train():
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    traindataset = TrainDataset()
    traindataloader = DataLoader(dataset = traindataset,batch_size=1,shuffle=False)
    testnet = TestNet().cuda()
    myloss = nn.MSELoss().cuda()
    optimizer = optim.SGD(testnet.parameters(), lr=0.001 )
    for epoch in range(100):
        for data,label in traindataloader :
            print("\n=====迭代开始=====")
            data = data.cuda()
            label = label.cuda()
            output = testnet(data)
            print("输入数据：",data)
            print("输出数据：",output)
            print("标签：",label)
            loss = myloss(output,label)
            optimizer.zero_grad()
            for name, parms in testnet.named_parameters():	
                print('-->name:', name)
                print('-->para:', parms)
                print('-->grad_requirs:',parms.requires_grad)
                print('-->grad_value:',parms.grad)
                print("===")
            loss.backward()
            optimizer.step()
            print("=============更新之后===========")
            for name, parms in testnet.named_parameters():	
                print('-->name:', name)
                print('-->para:', parms)
                print('-->grad_requirs:',parms.requires_grad)
                print('-->grad_value:',parms.grad)
                print("===")
            print(optimizer)
            for parms in testnet.parameters():
                print(parms)
            print(type(testnet.state_dict()))
            input("=====迭代结束=====")

            



if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(3)
    train()


