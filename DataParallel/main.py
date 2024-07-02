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
from torch.utils.data import DataLoader
import argparse
from torch.nn.parallel import DistributedDataParallel as DDP

import sys; sys.path.append("./model")
sys.path.append("./distributed_framework")
sys.path.append("./util_lib")
from model.ResNet import ResNet
from distributed_framework.PS import PS_BSP
from util_lib.dist_data_process import Data_Partition

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

parser = argparse.ArgumentParser(description='Test for argparse')
parser.add_argument('--Flag', '-f', help='flag of first time to run', default=False,type=bool,required=False)
parser.add_argument('--GPU_list', '-g', help='placement of each process:cudeID', required=True)
parser.add_argument('--Id_index', '-i', help='index for job (checkpoint)', required=True)
parser.add_argument('--Network', '-n', help='Network:18 34 50 101 152', required=True)
parser.add_argument('--Batch_size', '-b', help='batch_size:8 16 32 64', required=True)
parser.add_argument('--Epoch', '-e', help='epoch size:40 60 80', required=True)
parser.add_argument('--Datasets_dir', '-d', help='Datasets_dir', required=True)
parser.add_argument('--Worker_Num', '-w', help='Worker_Num', required=True)

def parse_devices(device_string):
    if device_string is None:
        return list(range(torch.cuda.device_count()))
    return [int(x) for x in device_string.split(',')]

args = parser.parse_args()

EPOCH=int(str(args.Epoch))
batch_size = int(str(args.Batch_size))
job_id = str(args.Id_index) #The index of job to find the right checkpoint
model_dir = "./check_point/" + job_id + "_model.pt"
epoch_dir = "./check_point/" + job_id + "_epoch.pt"
gpu_list = parse_devices(args.GPU_list)
datasets_list=Data_Partition(str(args.Datasets_dir),int(str(args.Worker_Num)))


model=ResNet(int(str(args.Network)),int(str(args.Batch_size)),args.Flag,model_dir,epoch_dir,EPOCH)


PS_BSP(model,datasets_list,gpu_list,int(str(args.Worker_Num)))

# 验证