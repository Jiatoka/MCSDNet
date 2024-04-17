import os
import sys
sys.path.append(os.getcwd())
from transformers import get_cosine_schedule_with_warmup,get_linear_schedule_with_warmup,get_polynomial_decay_schedule_with_warmup
import torch
from utils.utils import FocalLoss
def build_scheduler(scheduler_name:str,optimizer,num_warmup_steps:None,num_traning_steps:None):
    if scheduler_name=='get_linear_schedule_with_warmup':
        return get_linear_schedule_with_warmup(optimizer,num_warmup_steps,num_traning_steps)
    elif scheduler_name=='get_cosine_schedule_with_warmup':
        return get_cosine_schedule_with_warmup(optimizer,num_warmup_steps,num_traning_steps)
    elif scheduler_name=='get_polynomial_decay_schedule_with_warmup':
        return get_polynomial_decay_schedule_with_warmup(optimizer,num_warmup_steps,num_traning_steps,power=0.5)
    elif scheduler_name=='MultiStep':
        return torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer,milestones=[10,45,85],gamma=0.1)
    else:
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,patience=5)
def builid_optimizer(optimizer_name:str,model:torch.nn.Module,lr=0.001,weight_decay=0.0):
    if optimizer_name=='Adam':
        return torch.optim.Adam(lr=lr,params=model.parameters(),weight_decay=weight_decay)
    else:
        return torch.optim.SGD(lr=lr,params=model.parameters(),weight_decay=weight_decay)
def bulid_loss_fn(loss_name):
    if loss_name=='BCE':
        return torch.nn.BCEWithLogitsLoss()
    elif loss_name=='CE':
        return torch.nn.CrossEntropyLoss()
    else:
        return FocalLoss()

if __name__=='__main__':
    pass
    