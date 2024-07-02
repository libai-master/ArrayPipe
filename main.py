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
import argparse
import sys; sys.path.append("./model")
sys.path.append("./distributed_framework")
sys.path.append("./util_lib")
from model.VGG import Partition_Model_VGG
from distributed_framework.pipeline import Min_Pipeline
from distributed_framework.pipeline import Micro_Pipeline
from distributed_framework.pipeline import Arraypipe_Gpipe
from util_lib.data_process import Data_Process
from util_lib.data_process import Dataloader_Process
from util_lib.Job import Generate_job_stream
from util_lib.Schedule import Generate_scheduling


parser = argparse.ArgumentParser(description='Test for argparse')
parser.add_argument('--GPU_list', '-g', help='placement of each process:cudeID', required=True)
parser.add_argument('--Id_index', '-i', help='index for job (checkpoint)', required=True)
parser.add_argument('--Network', '-n', help='Network:11 13 16 19', required=True)
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
gpu_list = parse_devices(args.GPU_list)
dataset=Data_Process(str(args.Datasets_dir))

# partition_lists=Partition()
# Sub_Model_List=Partition_Model_VGG(partition_lists,batch_size,EPOCH)
# Min_Pipeline(Sub_Model_List,dataset,batch_size,gpu_list)

# Sub_Model_List=Partition_Model_VGG(partition_lists,batch_size,EPOCH,4)
# Micro_Pipeline(Sub_Model_List,dataset,batch_size,gpu_list,4)

job_stream=Generate_job_stream(100)
scheduling_list=Generate_scheduling()
train_dataloader_list=[]
N_data_list=[]
for job in job_stream:
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=job.batch_size, shuffle=True, num_workers=3,
                                               pin_memory=True)
    train_dataloader_list.append(train_dataloader)
    N_data_list=Dataloader_Process(train_dataloader_list,1500)

# print(N_data_list)


Arraypipe_Gpipe(scheduling_list,job_stream,dataset,gpu_list,N_data_list)
