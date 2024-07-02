from __future__ import print_function, division
import copy
from torchvision import datasets, transforms
from torch.utils.data import random_split


train_transforms = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomResizedCrop(224),  # 对图片尺寸做一个缩放切割
    transforms.RandomHorizontalFlip(),  # 水平翻转
    transforms.ToTensor(),  # 转化为张量
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # 进行归一化
])

def Data_Process(dataset_dir):
    train_datasets = datasets.ImageFolder(dataset_dir, transform=train_transforms)
    return train_datasets

def Dataloader_Process(dataloader_list,iteration_num):
    N_Data_list=[[] for _ in range(iteration_num)]
    for dataloader in dataloader_list:
        for idx,(x,y) in enumerate(dataloader):
            if idx==iteration_num:
                break
            else:
                N_Data_list[idx].append((x,y))
    return N_Data_list