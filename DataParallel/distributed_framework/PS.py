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
from tqdm import tqdm

from torch.utils.data import DataLoader
import datetime
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def PS_BSP(model,datasets_list,gpu_list,worker_num):
    env_dict = {
        key: os.environ[key]
        for key in ("MASTER_ADDR", "MASTER_PORT", "WORLD_SIZE", "LOCAL_WORLD_SIZE")
    }
    print(f"[{os.getpid()}] Initializing process group with: {env_dict}")
    dist.init_process_group(backend="gloo", timeout=datetime.timedelta(seconds=30)) # NCCL
    global_rank = int(os.environ["RANK"])

    if global_rank!=0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_list[global_rank])
        device=torch.device("cuda:0")
        model.to(device)
        dis_data_loader = DataLoader(datasets_list[global_rank], int(model.batchsize/worker_num), shuffle=True,num_workers=8,pin_memory=True) 
        for epoch in range(0, model.EPOCH):
            for batch_idx, (batch_x, batch_y) in enumerate(dis_data_loader):
                model.train_iteration(batch_x, batch_y,device)
                gradients_list=model.get_gradients()
                # time.sleep(1.1)
                for grad_layer in gradients_list:
                    dist.gather(tensor=grad_layer, dst = 0)

                for grad_layer in gradients_list:
                    temp_tensor=torch.zeros_like(grad_layer)
                    dist.scatter(temp_tensor,  src=0)
                    grad_layer=temp_tensor.cuda()

                model.load_gradients(gradients_list)
                model.step()

    elif global_rank==0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_list[global_rank])
        device=torch.device("cuda:0")
        model.to(device)
        dis_data_loader = DataLoader(datasets_list[global_rank], int(model.batchsize/worker_num), shuffle=True,num_workers=8,pin_memory=True) 
        for epoch in range(0, model.EPOCH):
            for batch_idx, (batch_x, batch_y) in enumerate(tqdm(dis_data_loader, desc="Iteration")):
                model.train_iteration(batch_x, batch_y,device)
                gradients_list=model.get_gradients()
                List=[]
                for grad_layer in gradients_list:
                    temp_list = [torch.zeros_like(grad_layer) for _ in range(worker_num)]
                    dist.gather(tensor=grad_layer, dst = 0, gather_list=temp_list)
                    temp_grad=temp_list[0]
                    for idx,grad in enumerate(temp_list):
                        if idx!=0:
                            grad=grad.cuda()
                            temp_grad=temp_grad+grad
                    List.append(temp_grad)

                model.zero_grad()
                model.load_gradients(List)
                model.step()

                for grad_layer in List:
                    temp_list=[]
                    for _ in range(worker_num):
                        temp_list.append(grad_layer)
                    temp_tensor=torch.zeros_like(grad_layer)
                    dist.scatter(temp_tensor, temp_list, src=0)
            
            model.save_model()
            model.save_epoch(epoch+1)


