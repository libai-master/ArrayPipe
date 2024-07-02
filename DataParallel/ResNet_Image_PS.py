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
# import torchsummary as summary
import os
import csv
import codecs
import numpy as np
import time
#from thop import profile
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from mpi4py import MPI
import argparse

parser = argparse.ArgumentParser(description='Test for argparse')
parser.add_argument('--Flag', '-f', help='flag of first time to run', required=True)
parser.add_argument('--GPU_list', '-g', help='placement of each process', required=True)
parser.add_argument('--Id_index', '-i', help='index for job (checkpoint)', required=True)
parser.add_argument('--Network', '-n', help='Network:18 34 50 101 152', required=True)
parser.add_argument('--Batch_size', '-b', help='batch_size:8 16 32 64', required=True)
parser.add_argument('--Epoch', '-e', help='epoch size:40 60 80', required=True)
args = parser.parse_args()
# EPOCH = 20
# if str(args.Epoch)=="10":
#     EPOCH = 10
# elif str(args.Epoch)=="15":
#     EPOCH = 15
# elif str(args.Epoch)=="20":
#     EPOCH = 20

EPOCH=int(str(args.Epoch))
def parse_devices(device_string):
    if device_string is None:
        return list(range(torch.cuda.device_count()))
    return [int(x) for x in device_string.split(',')]

job_id = str(args.Id_index) #The index of job to find the right checkpoint
model_dir = "/root/ExplSched/Scheduling/check_point/" + job_id + "_model.pt"
epoch_dir = "/root/ExplSched/Scheduling/check_point/" + job_id + "_epoch.pt"

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
'''定义超参数'''

batch_size = 8
if str(args.Batch_size)=="8":
    batch_size = 8
elif str(args.Batch_size)=="16":
    batch_size = 16
elif str(args.Batch_size)=="32":
    batch_size = 32
else:
    batch_size=64
classes_num = 1000
learning_rate = 1e-3
gpu_list = parse_devices(args.GPU_list)
first_run_flag = str(args.Flag)



COMM = MPI.COMM_WORLD
rank=COMM.Get_rank()
size=COMM.Get_size()
DEVICE = torch.device("cuda:1")
device=DEVICE
if gpu_list[rank] == 0:
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    device = torch.device('cuda:0')
elif gpu_list[rank] == 1:
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    device = torch.device('cuda:0')
elif gpu_list[rank] == 2:
    os.environ["CUDA_VISIBLE_DEVICES"] = '2'
    device = torch.device('cuda:0')
elif gpu_list[rank] == 3:
    os.environ["CUDA_VISIBLE_DEVICES"] = '3'
    device = torch.device('cuda:0')
elif gpu_list[rank] == 4:
    os.environ["CUDA_VISIBLE_DEVICES"] = '4'
    device = torch.device('cuda:0')
elif gpu_list[rank] == 5:
    os.environ["CUDA_VISIBLE_DEVICES"] = '5'
    device = torch.device('cuda:0')
elif gpu_list[rank] == 6:
    os.environ["CUDA_VISIBLE_DEVICES"] = '6'
    device = torch.device('cuda:0')
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = '7'
    device = torch.device('cuda:0')


'''定义Transform'''
# 对训练集做一个变换
train_transforms = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomResizedCrop(224),  # 对图片尺寸做一个缩放切割
    transforms.RandomHorizontalFlip(),  # 水平翻转
    transforms.ToTensor(),  # 转化为张量
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # 进行归一化
])
# 对测试集做变换
val_transforms = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

train_dir = "/root/mini-imagenet/train"  # 训练集路径
train_datasets = datasets.ImageFolder(train_dir, transform=train_transforms)

val_dir = "/root/mini-imagenet/val"
val_datasets = datasets.ImageFolder(val_dir, transform=val_transforms)


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

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=classes_num):
        super(ResNet, self).__init__()
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
        self.classifier = nn.Linear(512 * block.expansion, num_classes)  #

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

def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])

def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])

def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])

def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])

def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])


# --------------------训练过程---------------------------------
model = ResNet152()
if str(args.Network)=="18":
    model=ResNet18()
elif str(args.Network)=="34":
    model=ResNet34()
elif str(args.Network)=="50":
    model=ResNet50()
elif str(args.Network)=="101":
    model=ResNet101()
else:
    model=ResNet152()

  # 在这里更换你需要训练的模型
# model.to(device)
# summary.summary(model, input_size=(3, 224, 224), device='cuda')  # 我们选择图形的出入尺寸为(3,224,224)

params = [{'params': md.parameters()} for md in model.children()
          if md in [model.classifier]]
# optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
StepLR = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)  # 按训练批次调整学习率，每30个epoch调整一次
loss_func = nn.CrossEntropyLoss()
# 存储测试lo7g5  q23ss和acc
Loss_list = []
Accuracy_list = []
# 存储训练loss和acc
train_Loss_list = []
train_Accuracy_list = []
# 这俩作用是为了提前开辟一个
loss = []
loss1 = []

def distributed_samper(dataset, num_workers):
    dataset_list = []
    total_datasize = len(dataset)
    dataset_temp = copy.deepcopy(dataset)
    if total_datasize%num_workers!=0:
        dataset,abort=random_split(dataset=dataset,lengths=[int(total_datasize-total_datasize%num_workers),
                                                            total_datasize%num_workers])
        total_datasize = len(dataset)
        dataset_temp = copy.deepcopy(dataset)
    for i in range(num_workers):
        cuurent_size = len(dataset_temp)
        if cuurent_size == total_datasize / (num_workers):
            dataset_list.append(dataset_temp)
            break 
        dataset1, dataset2 = random_split(dataset=dataset_temp, lengths=[int(total_datasize / (num_workers)), int(cuurent_size - total_datasize / (num_workers))])
        dataset_list.append(dataset1)
        dataset_temp = dataset2
    return dataset_list

def grad_avg(g):
    #print(g[0])
    ret=copy.deepcopy(g[0])
    for i in range(len(ret)):
        for tt in range(1,len(g)):
            g[tt][i]=g[tt][i].cuda()
            ret[i]=ret[i].cuda()
            ret[i]+=g[tt][i]
        # ret[i] = torch.div(ret[i], len(g))
    return ret

def parse_devices(device_string):
    if device_string is None:
        return list(range(torch.cuda.device_count()))
    return [int(x) for x in device_string.split(',')]


def train_res(model, train_dataloader, epoch,t):
    start_time=time.time()
    model.train()
    # print('epoch {}'.format(epoch + 1))
    # training-----------------------------
    train_loss = 0.
    train_acc = 0.
    #print('data : {}'.format(len(train_dataloader)))
    # for batch_idx, (features, targets) in enumerate(train_loader):
    for batch_idx, (batch_x, batch_y) in enumerate(train_dataloader):
        batch_x = Variable(batch_x).to(device)
        batch_y = Variable(batch_y).to(device)
        optimizer.zero_grad()
        out = model(batch_x).to(device)
        loss = loss_func(out, batch_y)
        train_loss += loss.item()
        pred = torch.max(out, 1)[1]
        train_correct = (pred == batch_y).sum()
        train_acc += train_correct.item()
        loss.backward()
        optimizer.step()

        time_dir = "/root/ExplSched/Scheduling/time_loss/" + "ResNet"+str(args.Network)+"_"+job_id+"_"+str(rank)+"_time.txt"
        with open(time_dir,"a") as time_t:
            
            if batch_idx % 100 == 0:
                current_t = (time.time() - t) / 60
                epoch_t=(time.time()-start_time)/60
                time_loss=str(epoch)+' '+str(loss.item())+' '+str(current_t)+' '+str(epoch_t)
                time_t.write(time_loss)
                time_t.write('\n')
    #     # print(f'{batch_idx + 1:05d} iter takes {iter_end - iter_since:.4f}s')
    # print('Train Loss: {:.6f}, Acc: {:.6f}'.format(train_loss / (len(train_datasets)),
    #                                                train_acc / (len(train_datasets))))  # 输出训练时的loss和acc
    # train_Loss_list.append(train_loss / (len(val_datasets)))
    # train_Accuracy_list.append(100 * train_acc / (len(val_datasets)))


# evaluation--------------------------------
def val(model, val_dataloader):
    model.eval()
    eval_loss = 0.
    eval_acc = 0.
    for batch_idx, (batch_x, batch_y) in enumerate(val_dataloader):
        batch_x = Variable(batch_x, volatile=True).to(device)
        batch_y = Variable(batch_y, volatile=True).to(device)
        out = model(batch_x).to(device)
        loss = loss_func(out, batch_y)
        eval_loss += loss.item()
        pred = torch.max(out, 1)[1]
        num_correct = (pred == batch_y).sum()
        eval_acc += num_correct.item()
    
    Loss_list.append(eval_loss / (len(val_datasets)))
    Accuracy_list.append(100 * eval_acc / (len(val_datasets)))


# 保存模型的参数
# torch.save(model.state_dict(), 'ResNet18.pth')
# state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
# torch.save(state, 'ResNet18.pth')


log_dir = 'D:/2021year/CVPR/PermuteNet-main/Keras_position/Pytorch_Code/resnet18.pth'

BUFFSIZE=2**32
Mpi_buf = bytearray(BUFFSIZE)

if __name__ == '__main__':
    
    # if gpu_list[rank] == 0:
    #     os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    #     device = torch.device('cuda:0')
    # elif gpu_list[rank] == 1:
    #     os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    #     device = torch.device('cuda:0')
    # elif gpu_list[rank] == 2:
    #     os.environ["CUDA_VISIBLE_DEVICES"] = '2'
    #     device = torch.device('cuda:0')
    # else:
    #     os.environ["CUDA_VISIBLE_DEVICES"] = '3'
    #     device = torch.device('cuda:0')
    # model.to(device)

    dataset_train_list = distributed_samper(dataset=train_datasets, num_workers=(size-2)*2) 
    dataset_test_list = distributed_samper(dataset=val_datasets, num_workers=(size-2)*2)
    local_batch_size = int(batch_size/((size-2)*2))

    # if first_run_flag == "true": #if the flag is True, represent the MPI process is not the first time to run 
    #         #print("this code run")
        

    if rank!=0 and rank!=1:
        model.to(device)
        start_epoch=1
        if first_run_flag == "true": #if the flag is True, represent the MPI process is not the first time to run 
            #print("this code run")
            model.load_state_dict(torch.load(model_dir))
            start_epoch = torch.load(epoch_dir)
            start_epoch = start_epoch + 1
            print(start_epoch) 
        data_loader = DataLoader(dataset_train_list[rank-1], local_batch_size, shuffle=True,num_workers=8,pin_memory=True) 
        t=time.time()
        for epoch in range(start_epoch, EPOCH+1):
            train_res(model, data_loader, epoch,t)
            grad_list=[]
            for p in model.parameters():
                grad=p.grad
                grad_list.append(grad)
            MPI.Attach_buffer(Mpi_buf)
            COMM.bsend(grad_list,dest=1,tag=999)
            param_glob=COMM.recv(source=MPI.ANY_SOURCE,tag=MPI.ANY_TAG)
            MPI.Detach_buffer()
            model.load_state_dict(param_glob)
            val_dataloader=DataLoader(dataset_test_list[rank-1], local_batch_size, shuffle=True,num_workers=8,pin_memory=True) 
            val(model, val_dataloader)
            torch.save(epoch, epoch_dir, _use_new_zipfile_serialization=False)
            
    elif rank==1:
        model.to(device)
        if first_run_flag == "true": #if the flag is True, represent the MPI process is not the first time to run 
        #print("this code run")
            model.load_state_dict(torch.load(model_dir))

        while(1):
            optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
            StepLR = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
            MPI.Attach_buffer(Mpi_buf)
            g=[]
            for i in range(0,size-2):
                grad_list=COMM.recv(source=MPI.ANY_SOURCE,tag=999)
                g.append(grad_list)
            g=grad_avg(g)
            optimizer.zero_grad()
            for tmp_g,tmp_p in zip(g, model.named_parameters()):
                if tmp_g is not None:
                    tmp_p[1].grad = tmp_g
            optimizer.step()
            torch.save(model.state_dict(), model_dir)
            param_glob=model.state_dict()
            for i in range(2,size):
                COMM.bsend(param_glob,dest=i,tag=999)
            
            MPI.Detach_buffer()