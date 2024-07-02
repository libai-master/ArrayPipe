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
from util_lib.partition import Partition
from util_lib.swap import delete_param_grad_buf
from util_lib.swap import swap_in
from util_lib.swap import swap_out
from util_lib.swap import set_param_requiresgrad
from torch.utils.data import DataLoader
import datetime
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from multiprocessing import Process
import copy
from model.VGG import Sub_VGG
import psutil
from psutil._common import bytes2human

if os.environ.get('CUDA_LAUNCH_BLOCKING') in ['1','True', True]:
    MEMCPY_NONBLK = False
else:
    MEMCPY_NONBLK = True

class PrintCPUMem(object):
    """ NOTE: use with caution. High overhead. """
    def __init__(self, 
                train_loop_choices=[], # ["process","system"], 
                the_rest_choices=["system"]): # ["tensor","process","system"]
        self.p = psutil.Process(None) # None means self pid
        self.train_loop_choices = train_loop_choices
        self.the_rest_choices = the_rest_choices
        
    def tensor_cpu_memory(self):
        """ return current cpu tensors' total size, in current process """
        total_bytes = sum([obj.numel()*obj.element_size() for obj in gc.get_objects() 
                            if isinstance(obj, torch.Tensor) and not obj.is_cuda])
        return "cpu tensor %s"%bytes2human(total_bytes) # bytes int, return string

    def process_cpu_memory(self, metrics=["uss", "pss", "vms", "shared"]):
        
        named_tuple = self.p.memory_full_info() # pfullmem(rss=10199040, vms=52133888, shared=3887104, text=2867200, lib=0, data=5967872, dirty=0, uss=6545408, pss=6872064, swap=0)
        ps = []
        for name in metrics:
            value = getattr(named_tuple, name)
            value = bytes2human(value)
            ps.append('%s %s'%(name.capitalize(),value))
        return ', '.join(ps)

    def system_cpu_memory(self, metrics=["used", "shared", "available", "swap", "occupied"]):
        named_tuple = psutil.virtual_memory() # svmem(total=810210418688, available=801578504192, percent=1.1, used=3584335872, free=737506328576, active=25313386496, inactive=41244495872, buffers=4782653440, cached=64337100800, shared=2789376, slab=3403677696)
        named_tuple2 = psutil.swap_memory() # sswap(total=1023406080, used=0, free=1023406080, percent=0.0, sin=0, sout=0)
        ps = []
        for name in metrics:
            if name == "swap":
                value = getattr(named_tuple2, 'used')
            elif name == "occupied":
                value = getattr(named_tuple, 'total') - getattr(named_tuple, 'available')
            else:
                value = getattr(named_tuple, name)
            #
            if name != 'percent':
                value = bytes2human(value)
            ps.append('%s %s'%(name.capitalize(),value))
        return ', '.join(ps)

    def print(self, title="", train_loop=False):
        choices = self.train_loop_choices if train_loop else self.the_rest_choices
        if choices == []:
            return
        ps = [title]
        for c in choices:
            if c == "tensor":
                ps.append(self.tensor_cpu_memory())
            elif c == "process":
                ps.append(self.process_cpu_memory())
            elif c == "system":
                ps.append(self.system_cpu_memory())
        print(" | ".join(ps))
        # print(title, "|", tensor_cpu_memory(), "|", process_cpu_memory(metrics=["uss", "pss", "vms", "shared", "swap"]), "|", system_cpu_memory(metrics=["used", "shared", "available", "swap"]))

# def dump(obj, path):
#     if ".pickle" not in path:
#         path += ".pickle"
#     with open(path, 'wb') as f:
#         # pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
#         torch.save(obj, f)



def Min_Pipeline(Sub_Model_List,dataset,batch_size,gpu_list):
    env_dict = {
        key: os.environ[key]
        for key in ("MASTER_ADDR", "MASTER_PORT", "WORLD_SIZE", "LOCAL_WORLD_SIZE")
    }
    print(f"[{os.getpid()}] Initializing process group with: {env_dict}")
    dist.init_process_group(backend="NCCL", timeout=datetime.timedelta(seconds=30)) # NCCL
    global_rank = int(os.environ["RANK"])
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8,
                                               pin_memory=True)
    if global_rank==0:
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_list[global_rank])
        device=torch.device("cuda:{}".format(str(gpu_list[global_rank])))
        sub_model=Sub_Model_List[0]
        sub_model.to(device)
        for epoch in range(0, sub_model.EPOCH):
            for batch_idx, (batch_x, batch_y) in enumerate(tqdm(train_dataloader)):
                out_y=sub_model.forward(batch_x,batch_y,device)
                # print(type(out_y))
                # print('global_rank:{},send y'.format(global_rank))
                # print(out_y.shape)
                dist.send(tensor=out_y,dst=1,tag=0)
                grad_y=torch.zeros_like(out_y).to(device)
                dist.recv(tensor=grad_y,src=1,tag=1)
                # print(grad_y.shape)
                sub_model.backward(grad_y,device)
                sub_model.step()

    elif global_rank==len(gpu_list)-1:
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_list[global_rank])
        device=torch.device("cuda:{}".format(str(gpu_list[global_rank])))
        sub_model=Sub_Model_List[global_rank]
        sub_model.to(device)
        for epoch in range(0, sub_model.EPOCH):
            for batch_idx, (batch_x, batch_y) in enumerate(train_dataloader):
                # shape=[batch_size,]
                out_x=torch.zeros([batch_size, 512, 14, 14]).to(device)  
                dist.recv(tensor=out_x,src=global_rank-1,tag=global_rank-1)  
                # print('global_rank:{},recv x'.format(global_rank))
                # print(out_x.shape)
                out_y=sub_model.forward(out_x,batch_y,device)
                # print(out_y.shape)
                sub_model.backward()
                # print('global_rank:{},send grad'.format(global_rank))
                # print(sub_model.batch_x.grad.shape)
                dist.send(tensor=sub_model.batch_x.grad,dst=global_rank-1,tag=global_rank)
                sub_model.step()
    else:
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_list[global_rank])
        device=torch.device("cuda:{}".format(str(gpu_list[global_rank])))
        sub_model=Sub_Model_List[global_rank]
        sub_model.to(device)
        for epoch in range(0, sub_model.EPOCH):
            for batch_idx, (batch_x, batch_y) in enumerate(train_dataloader):
                if global_rank==1:
                    out_x=torch.zeros(batch_size, 128, 56, 56).to(device)
                if global_rank==2:
                    out_x=torch.zeros(batch_size, 256, 28, 28).to(device)
                dist.recv(tensor=out_x,src=global_rank-1,tag=global_rank-1) 
                # print('global_rank:{},recv x'.format(global_rank))
                # print(out_x.shape)
                out_y=sub_model.forward(out_x,batch_y,device)
                # print('global_rank:{},send y '.format(global_rank))
                # print(out_y.shape)
                dist.send(tensor=out_y,dst=global_rank+1,tag=global_rank)

                grad_y=torch.zeros_like(out_y).to(device)
                dist.recv(tensor=grad_y,src=global_rank+1,tag=global_rank+1)
                sub_model.backward(grad_y,device)
                dist.send(tensor=sub_model.batch_x.grad,dst=global_rank-1,tag=global_rank)
                sub_model.step()

def Micro_Pipeline(Sub_Model_List,dataset,batch_size,gpu_list,Micro_number):
    env_dict = {
        key: os.environ[key]
        for key in ("MASTER_ADDR", "MASTER_PORT", "WORLD_SIZE", "LOCAL_WORLD_SIZE")
    }
    batch_size=batch_size/Micro_number
    print(f"[{os.getpid()}] Initializing process group with: {env_dict}")
    dist.init_process_group(backend="NCCL", timeout=datetime.timedelta(seconds=30)) # NCCL
    global_rank = int(os.environ["RANK"])
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8,
                                               pin_memory=True)
    if global_rank==0:
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_list[global_rank])
        device=torch.device("cuda:{}".format(str(gpu_list[global_rank])))
        sub_model=Sub_Model_List[0]
        sub_model.to(device)
        for epoch in range(0, sub_model.EPOCH):
            for _ in range(Micro_number):
                for batch_idx, (batch_x, batch_y) in enumerate(tqdm(train_dataloader)):
                    out_y=sub_model.forward(batch_x,batch_y,device)
                    # print(type(out_y))
                    # print('global_rank:{},send y'.format(global_rank))
                    # print(out_y.shape)
                    dist.send(tensor=out_y,dst=1,tag=0)
            for _ in range(Micro_number):
                grad_y=torch.zeros_like(out_y).to(device)
                dist.recv(tensor=grad_y,src=1,tag=1)
                # print(grad_y.shape)
                sub_model.Micro_backward_step(grad_y,device)


    elif global_rank==len(gpu_list)-1:
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_list[global_rank])
        device=torch.device("cuda:{}".format(str(gpu_list[global_rank])))
        sub_model=Sub_Model_List[global_rank]
        sub_model.to(device)
        for epoch in range(0, sub_model.EPOCH):
            for _ in range(Micro_number):
                for batch_idx, (batch_x, batch_y) in enumerate(train_dataloader):
                    # shape=[batch_size,]
                    out_x=torch.zeros([batch_size, 512, 14, 14]).to(device)  
                    dist.recv(tensor=out_x,src=global_rank-1,tag=global_rank-1)  
                    # print('global_rank:{},recv x'.format(global_rank))
                    # print(out_x.shape)
                    out_y=sub_model.forward(out_x,batch_y,device)
            for _ in range(Micro_number):
                # print(out_y.shape)
                sub_model.Micro_backward_step()
                # print('global_rank:{},send grad'.format(global_rank))
                # print(sub_model.batch_x.grad.shape)
                dist.send(tensor=sub_model.batch_x.grad,dst=global_rank-1,tag=global_rank)
                
    else:
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_list[global_rank])
        device=torch.device("cuda:{}".format(str(gpu_list[global_rank])))
        sub_model=Sub_Model_List[global_rank]
        sub_model.to(device)
        for epoch in range(0, sub_model.EPOCH):
            for batch_idx, (batch_x, batch_y) in enumerate(train_dataloader):
                if global_rank==1:
                    out_x=torch.zeros(batch_size, 128, 56, 56).to(device)
                if global_rank==2:
                    out_x=torch.zeros(batch_size, 256, 28, 28).to(device)
                dist.recv(tensor=out_x,src=global_rank-1,tag=global_rank-1) 
                # print('global_rank:{},recv x'.format(global_rank))
                # print(out_x.shape)
                out_y=sub_model.forward(out_x,batch_y,device)
                # print('global_rank:{},send y '.format(global_rank))
                # print(out_y.shape)
                dist.send(tensor=out_y,dst=global_rank+1,tag=global_rank)

                grad_y=torch.zeros_like(out_y).to(device)
                dist.recv(tensor=grad_y,src=global_rank+1,tag=global_rank+1)
                sub_model.backward(grad_y,device)
                dist.send(tensor=sub_model.batch_x.grad,dst=global_rank-1,tag=global_rank)
                sub_model.step()






def Arraypipe_Gpipe(scheduling_list,job_stream,dataset,gpu_list,N_data_list):
    env_dict = {
        key: os.environ[key]
        for key in ("MASTER_ADDR", "MASTER_PORT", "WORLD_SIZE", "LOCAL_WORLD_SIZE")
    }
    print(f"[{os.getpid()}] Initializing process group with: {env_dict}")
    dist.init_process_group(backend="NCCL", timeout=datetime.timedelta(seconds=30)) # NCCL
    global_rank = int(os.environ["RANK"])

    
    for N_xdata in N_data_list:
     
        if global_rank==0:

            compute_graph=scheduling_list[global_rank]
            device=torch.device("cuda:{}".format(str(gpu_list[global_rank])))
            sub_model=job_stream[0].Sub_Model_List[0]
            sub_model.to(device)
            for unit in compute_graph:
                
                # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_list[global_rank])            
                sub_model=job_stream[unit[0]].Sub_Model_List[0]                
                swap_in(sub_model.model,sub_model.cpu_model,device) 
                set_param_requiresgrad(sub_model.model)
                batch_x,batch_y=N_xdata[unit[0]]

                if unit[1]=='f':
                    out_y=sub_model.forward(batch_x,batch_y,device)
                    # print(type(out_y))
                    # print('global_rank:{},send y'.format(global_rank))
                    # print(out_y.shape)
                    dist.send(tensor=out_y,dst=1,tag=0)
                elif unit[1]=='b':
                    
                    grad_y=torch.zeros(job_stream[unit[0]].batch_size, 128, 56, 56).to(device)
                    dist.recv(tensor=grad_y,src=1,tag=1)
                    # print(grad_y.shape)
                    sub_model.backward(device,grad_y)
                    sub_model.step()
                
                swap_out(sub_model.model,sub_model.cpu_model)

                delete_param_grad_buf(sub_model.model)


        elif global_rank==len(gpu_list)-1:
            compute_graph=scheduling_list[global_rank]
            device=torch.device("cuda:{}".format(str(gpu_list[global_rank])))
            sub_model=job_stream[0].Sub_Model_List[global_rank]
            sub_model.to(device)
            for unit in compute_graph:
                sub_model=job_stream[unit[0]].Sub_Model_List[global_rank]
        
                swap_in(sub_model.model,sub_model.cpu_model,device)
                set_param_requiresgrad(sub_model.model)
                batch_x,batch_y=N_xdata[unit[0]]
                if unit[1]=='f':
                    out_x=torch.zeros([job_stream[unit[0]].batch_size, 512, 14, 14]).to(device)  
                    dist.recv(tensor=out_x,src=global_rank-1,tag=global_rank-1)  
                    # print('global_rank:{},recv x'.format(global_rank))
                    # print(out_x.shape)
                    out_y=sub_model.forward(out_x,batch_y,device)
                elif unit[1]=='b':
                    sub_model.backward(device)
                    # print('global_rank:{},send grad'.format(global_rank))
                    # print(sub_model.batch_x.grad.shape)
                    dist.send(tensor=sub_model.batch_x.grad,dst=global_rank-1,tag=global_rank)
                    sub_model.step()
                swap_out(sub_model.model,sub_model.cpu_model)
                delete_param_grad_buf(sub_model.model)
                
                    
        else:
            compute_graph=scheduling_list[global_rank]
            device=torch.device("cuda:{}".format(str(gpu_list[global_rank])))
            sub_model=job_stream[0].Sub_Model_List[global_rank]
            sub_model.to(device)
            for unit in compute_graph:
                # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_list[global_rank])   
                sub_model=job_stream[unit[0]].Sub_Model_List[global_rank]
                
                swap_in(sub_model.model,sub_model.cpu_model,device)
                set_param_requiresgrad(sub_model.model)
                batch_x,batch_y=N_xdata[unit[0]]
                if unit[1]=='f':
                    if global_rank==1:
                        out_x=torch.zeros(job_stream[unit[0]].batch_size, 128, 56, 56).to(device)
                    if global_rank==2:
                        out_x=torch.zeros(job_stream[unit[0]].batch_size, 256, 28, 28).to(device)
                    dist.recv(tensor=out_x,src=global_rank-1,tag=global_rank-1) 
                    # print('global_rank:{},recv x'.format(global_rank))
                    # print(out_x.shape)
                    out_y=sub_model.forward(out_x,batch_y,device)
                    dist.send(tensor=out_y,dst=global_rank+1,tag=global_rank)    
                elif unit[1]=='b':
                    # print('global_rank:{},send y '.format(global_rank))
                    # print(out_y.shape)
                    if global_rank==1:
                        grad_y=torch.zeros(job_stream[unit[0]].batch_size, 256, 28, 28).to(device)
                    if global_rank==2:
                        grad_y=torch.zeros([job_stream[unit[0]].batch_size, 512, 14, 14]).to(device)
                    dist.recv(tensor=grad_y,src=global_rank+1,tag=global_rank+1)
                    # print("xxxxxxx{},,,,{}".format(global_rank,unit[0]))
                    sub_model.backward(device,grad_y)
                    dist.send(tensor=sub_model.batch_x.grad,dst=global_rank-1,tag=global_rank)
                    sub_model.step()
                swap_out(sub_model.model,sub_model.cpu_model)
                delete_param_grad_buf(sub_model.model)
               