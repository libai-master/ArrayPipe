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
from torch.utils.data import DataLoader
import datetime
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from multiprocessing import Process
import copy
from model.VGG import Sub_VGG
import psutil
from psutil._common import bytes2human
import gc

if os.environ.get('CUDA_LAUNCH_BLOCKING') in ['1','True', True]:
    MEMCPY_NONBLK = False
else:
    MEMCPY_NONBLK = True

def delete_param_grad_buf(top_module, manual_gc=False):    
    def fn(m): # m = each module
        for key, param in m._parameters.items(): # OrderedDict([('weight', Parameter), ('bias', Parameter)])
            if param is not None:
                # delete param
                param.data = torch.empty(0, device="cpu")
                # delete grad
                if param.grad is not None:
                    param.grad = None
                param.detach_() # in-place detaches self Tensor from the graph that created it, making it a leaf.
                assert not param.requires_grad
        for key, buf in m._buffers.items(): # OrderedDict([('running_mean', tensor), ('running_var', tensor), ('num_batches_tracked', tensor)])
            if buf is not None:
                m._buffers[key] = torch.empty(0, device="cpu")
    top_module.apply(fn) # Applies ``fn`` recursively to every submodule (as returned by ``.children()`` as well as self.
        #
    if manual_gc:
        gc.collect(); torch.cuda.empty_cache() 

def swap_out(model,cpu_model):
    for gpu_m, cpu_m in zip(model.modules(), cpu_model.modules()): # iterator over all modules in the network
            for key, param in gpu_m._parameters.items(): # OrderedDict([('weight', Parameter), ('bias', Parameter)])
                if param is not None:
                    cpu_m._parameters[key].data.copy_(gpu_m._parameters[key].data,non_blocking=MEMCPY_NONBLK)  
                
            for key, buf in gpu_m._buffers.items(): # OrderedDict([('running_mean', tensor), ('running_var', tensor), ('num_batches_tracked', tensor)])
                if buf is not None:
                    # if len(buf.shape) == 0: # Scalar buffer stays on CPU
                    #     gpu_m._buffers[key] = buf.clone().detach().requires_grad_(False)
                    # else:
                        cpu_m._buffers[key].copy_(buf,non_blocking=MEMCPY_NONBLK) 

    model=cpu_model

def swap_in(model,cpu_model,DEVICE):
     for gpu_m, cpu_m in zip(model.modules(), cpu_model.modules()): # iterator over all modules in the network
            for key, param in cpu_m._parameters.items(): # OrderedDict([('weight', Parameter), ('bias', Parameter)])
                if param is not None:
                    gpu_m._parameters[key].data=param.to(DEVICE,non_blocking=MEMCPY_NONBLK) 
                
            for key, buf in cpu_m._buffers.items(): # OrderedDict([('running_mean', tensor), ('running_var', tensor), ('num_batches_tracked', tensor)])
                if buf is not None:
                        gpu_m._buffers[key].data= buf.to(DEVICE,non_blocking=MEMCPY_NONBLK)
def set_param_requiresgrad(model):
    for gpu_m in model.modules(): # iterator over all modules in the network
                    for key, param in gpu_m._parameters.items(): # OrderedDict([('weight', Parameter), ('bias', Parameter)])
                        if param is not None:
                            param.requires_grad_(True)