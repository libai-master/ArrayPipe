from __future__ import print_function, division
import copy
from torchvision import datasets, transforms
from torch.utils.data import random_split

'''定义Transform'''
# 对训练集做一个变换
train_transforms = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomResizedCrop(224),  # 对图片尺寸做一个缩放切割
    transforms.RandomHorizontalFlip(),  # 水平翻转
    transforms.ToTensor(),  # 转化为张量
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # 进行归一化
])


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

def Data_Partition(dataset_dir,num_workers):
    train_datasets = datasets.ImageFolder(dataset_dir, transform=train_transforms)
    return distributed_samper(train_datasets,num_workers)