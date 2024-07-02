from __future__ import print_function, division
import torch.nn.functional as F
import torch
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
import summary
#from thop import profile

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
'''定义超参数'''
EPOCH = 20
batch_size = 64
classes_num = 1000
learning_rate = 1e-3

DEVICE = torch.device("cuda:1")

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
# 定义数据集
train_datasets = datasets.ImageFolder(train_dir, transform=train_transforms)
# 加载数据集
train_dataloader = torch.utils.data.DataLoader(train_datasets, batch_size=batch_size, shuffle=True, num_workers=8,
                                               pin_memory=True)  # ,num_workers=16,pin_memory=False

val_dir = "/root/mini-imagenet/val"
val_datasets = datasets.ImageFolder(val_dir, transform=val_transforms)
val_dataloader = torch.utils.data.DataLoader(val_datasets, batch_size=batch_size, shuffle=True, num_workers=8,
                                             pin_memory=True)  # ,num_workers=16,pin_memory=True


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
model = ResNet152()  # 在这里更换你需要训练的模型
model = model.to(DEVICE)
summary.summary(model, input_size=(3, 224, 224), device='cuda')  # 我们选择图形的出入尺寸为(3,224,224)

params = [{'params': md.parameters()} for md in model.children()
          if md in [model.classifier]]
# optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
StepLR = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)  # 按训练批次调整学习率，每30个epoch调整一次
loss_func = nn.CrossEntropyLoss()
# 存储测试loss和acc
Loss_list = []
Accuracy_list = []
# 存储训练loss和acc
train_Loss_list = []
train_Accuracy_list = []
# 这俩作用是为了提前开辟一个
loss = []
loss1 = []


def train_res(model, train_dataloader, epoch):
    model.train()
    # print('epoch {}'.format(epoch + 1))
    # training-----------------------------
    train_loss = 0.
    train_acc = 0.
    print('data : {}'.format(len(train_dataloader)))
    # for batch_idx, (features, targets) in enumerate(train_loader):
    for batch_idx, (batch_x, batch_y) in enumerate(train_dataloader):
        iter_since = time.time()
        batch_x = Variable(batch_x).cuda()
        batch_y = Variable(batch_y).cuda()
        optimizer.zero_grad()
        out = model(batch_x)
        loss = loss_func(out, batch_y)
        train_loss += loss.item()
        pred = torch.max(out, 1)[1]
        train_correct = (pred == batch_y).sum()
        train_acc += train_correct.item()
        loss.backward()
        optimizer.step()
        iter_end = time.time()
        # print(f'{batch_idx + 1:05d} iter takes {iter_end - iter_since:.4f}s')
    print('Train Loss: {:.6f}, Acc: {:.6f}'.format(train_loss / (len(train_datasets)),
                                                   train_acc / (len(train_datasets))))  # 输出训练时的loss和acc
    train_Loss_list.append(train_loss / (len(val_datasets)))
    train_Accuracy_list.append(100 * train_acc / (len(val_datasets)))


# evaluation--------------------------------
def val(model, val_dataloader):
    model.eval()
    eval_loss = 0.
    eval_acc = 0.
    for batch_idx, (batch_x, batch_y) in enumerate(val_dataloader):
        batch_x = Variable(batch_x, volatile=True).cuda()
        batch_y = Variable(batch_y, volatile=True).cuda()
        out = model(batch_x)
        loss = loss_func(out, batch_y)
        eval_loss += loss.item()
        pred = torch.max(out, 1)[1]
        num_correct = (pred == batch_y).sum()
        eval_acc += num_correct.item()
    print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(val_datasets)),
                                                  eval_acc / (len(val_datasets))))  # 输出测试时的loss和acc
    Loss_list.append(eval_loss / (len(val_datasets)))
    Accuracy_list.append(100 * eval_acc / (len(val_datasets)))


# 保存模型的参数
# torch.save(model.state_dict(), 'ResNet18.pth')
# state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
# torch.save(state, 'ResNet18.pth')


log_dir = 'D:/2021year/CVPR/PermuteNet-main/Keras_position/Pytorch_Code/resnet18.pth'


def main():
    if torch.cuda.is_available():
        model.cuda()
    test_flag = False
    # 如果test_flag=True,则加载已保存的模型
    if test_flag:
        # 加载保存的模型直接进行测试机验证，不进行此模块以后的步骤
        checkpoint = torch.load(log_dir)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        val(model, val_dataloader)
        # 如果只评估模型，则保留这个return，如果是从某个阶段开始继续训练模型，则去掉这个模型
        # 同时把上面的False改成True
        return

    # 如果有保存的模型，则加载模型，并在其基础上继续训练
    if os.path.exists(log_dir) and test_flag:
        checkpoint = torch.load(log_dir)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        print('加载 epoch {} 成功！'.format(start_epoch))
    else:
        start_epoch = 1
        print('无保存模型，将从头开始训练！')

    for epoch in range(start_epoch, EPOCH):
        since = time.time()
        print('epoch {}'.format(epoch))  # 显示每次训练次数
        train_res(model, train_dataloader, epoch)
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))  # 输出训练和测试的时间
        # 通过一个if语句判断，让模型每十次评估一次模型并且保存一次模型参数
        epoch_num = epoch / 10
        epoch_numcl = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 15.0, 16.0, 17.0, 18.0,
                       19.0, 20.0, 21.0]
        print('epoch_num', epoch_num)
        if epoch_num in epoch_numcl:
            print('评估模型')
            val(model, val_dataloader)
            print('保存模型')
            state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
            torch.save(state, log_dir)


if __name__ == '__main__':
    main()

