import argparse
from argparse import Namespace
import datetime
import json
import numpy as np
import pickle 
import pandas as pd
import torch
import random
from tqdm import tqdm
from scipy.signal import resample
import os
import sys
from typing import Callable
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim.lr_scheduler import LambdaLR
from Configs import ARGS

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# CUDA_LAUNCH_BLOCKING=1
# os.environ["CUDA_LAUNCH_BLOCKING"]="1"
# os.environ["CUDA_VISIBLE_DEVICES"]="6"
from transformers import get_linear_schedule_with_warmup
import time
from LC_model import MYModel
from utils import get_classification_metrics,get_classification_metrics2,get_classification_metrics3
from utils import Summary,EarlyStopper
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader,Dataset
from dataLoader_my import MyDataset1_noscaler,ImbalancedDatasetSampler
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import warnings
import torch.nn.functional as F

import shutil
warnings.filterwarnings("ignore")
parser=argparse.ArgumentParser()

parser.add_argument("--model_name",default="gpt_allzi",type=str,help="")
parser.add_argument("--seq_len",default=512,type=int)

parser.add_argument("--p2n_rate_train",default=-1,type=float,help="num pos : num neg")
parser.add_argument("--p2n_rate_test",default=1,type=float,help="num pos : num neg")
parser.add_argument("--batch_size",default=128,type=int)
parser.add_argument("--llm_model",default="GPT2",help="llm model: or  GPT2 or Llama")
parser.add_argument("--tslm_model",default="GPT2",help="llm model: or  GPT2 or Llama")
parser.add_argument("--n_runs",default=5,type=int)


parser.add_argument("--timestamp_type",default="fixed",type=str,help="fixed,period,normal")
parser.add_argument("--ifLoRA",action="store_true")
parser.add_argument("--ifP_tuning",action="store_true") 
parser.add_argument("--out_proj",default="First",type=str)
parser.add_argument("--if_addtoken_grad",action="store_true")
parser.add_argument("--if_only_metavec",action="store_true")
parser.add_argument("--ifAddNoiseFlare",action="store_true")
parser.add_argument("--if_use_id",action="store_true")
parser.add_argument("--ifNorm",action="store_true")

parser.add_argument("--trend_period_extrac_type",default="wavelets",type=str,help="wavelets; my ")
parser.add_argument("--flare_process_type",default="FE",type=str,help="PA [patch add]; FA [as BL2]; FE [flare&residual]")
parser.add_argument("--id_process_type",default="uniq",type=str,help="uniq ; same ")


args=parser.parse_args()
old_args=ARGS()
old_args._update_class_variables(args)
args=old_args

start_wall_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
if torch.cuda.is_available():
    device=torch.device("cuda")
else:
    device=torch.device("cpu")
args.start_wall_time=start_wall_time
args.device=device


def seed_everything(seed=11):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        n_class = pred.size(1)
        eps = self.smoothing
        one_hot = torch.zeros_like(pred).scatter(1, target.unsqueeze(1), 1)
        one_hot = one_hot * (1 - eps) + eps / n_class
        log_prb = nn.functional.log_softmax(pred, dim=1)
        loss = -one_hot * log_prb
        return loss.sum(dim=1).mean()

def adjust_learning_rate(optimizer, epoch, lr_):
    lr_adjust = {epoch: lr_ * (0.5 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))




def train_one_epoch(train_loader,model,device,train_logger,optimizer,lr_scheduler,ifschedule_with_warmup,schedule_type):
    num_=0
    losses,probs,labels=[],[],[]
    for lc_time,lc_flux,lc_label,lc_mask,label,meta_feats,kid_name,lc_id in tqdm(train_loader, total=len(train_loader)):
        optimizer.zero_grad(set_to_none=True)

        lc_time=lc_time.float().to(device)
        lc_flux=lc_flux.float().to(device)
        lc_label=lc_label.long().to(device)
        lc_mask=lc_mask.float().to(device)
        label = label.long().to(device)
        meta_feats=meta_feats.float().to(device)
        lc_id=lc_id.long().to(device)

        predictions =model(lc_time,lc_flux,lc_label,lc_mask,label,meta_feats,kid_name,lc_id)
        B,L=lc_flux.shape
        loss = criterion(predictions , label)
        num_+=B
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if ifschedule_with_warmup and schedule_type!="ExponentialLR":
            lr_scheduler.step()

        losses.append(loss.item()*B)
        probs.append(F.softmax(predictions).detach().cpu().numpy()[:,1])
        labels.append(label.detach().cpu().numpy())
    
    if ifschedule_with_warmup and schedule_type=="ExponentialLR":
        lr_scheduler.step()

    average_loss = sum(losses)/num_
    train_logger.update_loss(average_loss)    
    probs = np.concatenate(probs, axis=0)
    labels = np.concatenate(labels, axis=0)
    return average_loss,probs,labels


def test_one_epoch(test_loader,model,device,test_logger):
    num_=0
    losses,probs,labels=[],[],[]
    model.eval()
    with torch.no_grad():
        for lc_time,lc_flux,lc_label,lc_mask,label,meta_feats,kid_name,lc_id in tqdm(test_loader, total=len(test_loader)):
            lc_time=lc_time.float().to(device)
            lc_flux=lc_flux.float().to(device)
            lc_label=lc_label.long().to(device)
            lc_mask=lc_mask.float().to(device)
            label = label.long().to(device)
            meta_feats=meta_feats.float().to(device)
            lc_id=lc_id.long().to(device)

            predictions =model(lc_time,lc_flux,lc_label,lc_mask,label,meta_feats,kid_name,lc_id)

            B,L=lc_flux.shape
            num_+=B
            loss = criterion(predictions, label)
            losses.append(loss.item()*B)
            probs.append(F.softmax(predictions).detach().cpu().numpy()[:,1])
            labels.append(label.detach().cpu().numpy())
    
    average_loss = sum(losses)/num_
    model.train()
    test_logger.update_loss(average_loss)
    probs = np.concatenate(probs, axis=0)
    labels = np.concatenate(labels, axis=0)
    return average_loss,probs,labels


if __name__=="__main__":

    seed=args.seed
    seed_everything(seed)

    model_name=args.model_name
    patience=args.patience
    MAX_epoches=args.MAX_epoches
    n_runs=args.n_runs
    npy_dir_path=args.npy_dir_path
    meta_feats_path=args.meta_feats_path
    kids_path=args.kids_path
    p2n_rate_train=args.p2n_rate_train
    p2n_rate_test=args.p2n_rate_test
    batch_size=args.batch_size

    # dim_hid=args.dim_hid

    dir_path=os.path.join(args.dir_path,f"{start_wall_time}")
    os.makedirs(dir_path,exist_ok=True)

    scores_file_path=os.path.join(dir_path,"scores.txt")

    # 打开日志文件进行写入
    log_file = open(os.path.join(dir_path,'output.log'), 'w')
    sys.stdout = log_file
    sys.stderr = log_file

    scoresss=[]
    names=[ 'auc', 'f1_score_macro', 'acc',  'f1_score_1',  'recall_1','precision_1',  'thres']
    with open(scores_file_path,"a",encoding="utf-8") as f:
        f.write(", ".join(names)+"\n")

    train_dataset=MyDataset1_noscaler(npy_dir_path,meta_feats_path,kids_path, p2n_rate_train,p2n_rate_test,data_split="train",if_only_metavec=args.if_only_metavec)
    train_loader  = DataLoader(train_dataset, sampler=ImbalancedDatasetSampler(train_dataset,num_samples=200000) ,batch_size=batch_size)
    test_dataset=MyDataset1_noscaler(npy_dir_path,meta_feats_path,kids_path, p2n_rate_train,p2n_rate_test,data_split="test",if_only_metavec=args.if_only_metavec)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    for splited_id in range(n_runs):
        train_logger=Summary(dir_path,task_name=f"train_{splited_id}")
        test_logger=Summary(dir_path,task_name=f"test_{splited_id}")
        tmp_dir_path=os.path.join(dir_path,f"__{splited_id}")
        os.makedirs(tmp_dir_path,exist_ok=True)
        stopper=EarlyStopper(tmp_dir_path,patience=patience)
    
        cur_epoch = 0
        max_epoch = MAX_epoches
        model=MYModel(model_name,args)
        print(f"Size of model : ", f"{sum(p.numel() for p in model.parameters() if p.requires_grad) / 10 ** 6:.3f}M")
        
        # criterion = torch.nn.CrossEntropyLoss()
        criterion = LabelSmoothingLoss(smoothing=0.1)


        criterion = criterion.to(device)

        model = model.to(device)
        optimizer = optim.AdamW(model.parameters(),lr=args.learning_rate,eps=args.adam_epsilon)

        if args.schedule_type=="ExponentialLR":
            lr_scheduler = ExponentialLR(optimizer, gamma=0.99)  # 每个epoch乘以0.9
        elif args.schedule_type=="LambdaLR":
            def lr_lambda(epoch):
                return (1 - epoch / (len(train_loader) * max_epoch)) ** 2  # 选择幂次来控制衰减速率
            lr_scheduler = LambdaLR(optimizer, lr_lambda)  
        else:
            lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=(len(train_loader) * max_epoch),
        )

        while cur_epoch < max_epoch:
            average_loss,probs,labels=train_one_epoch(train_loader,model,device,train_logger,optimizer,lr_scheduler,args.ifschedule_with_warmup,args.schedule_type)
            print(f"Epoch {cur_epoch}: ")

            thres=get_classification_metrics2(labels,probs)
            metrics=get_classification_metrics(labels,probs,thres=thres)
            train_logger.update_metrics([getattr(metrics,i) for i in names], names)
            tmp_out="".join([f"TRAIN {i}:{getattr(metrics,i):.3f} | " for i in names])
            print(f"Epoch {cur_epoch}: {tmp_out} | Train loss: {average_loss:.3f}")
            

            average_loss,probs,labels=test_one_epoch(test_loader,model,device,test_logger)
            thres=get_classification_metrics2(labels,probs)
            metrics=get_classification_metrics(labels,probs,thres=thres)
            test_logger.update_metrics([getattr(metrics,i) for i in names], names)
            tmp_out="".join([f"TEST {i}:{getattr(metrics,i):.3f} | " for i in names])
            print(f"Epoch {cur_epoch}: {tmp_out} | Test loss: {average_loss:.3f}")

            stopper.step(average_loss,cur_epoch,model,thres)
            if stopper.early_stop:
                break
            cur_epoch += 1
            
        
        stopper.load_checkpoint(model)

        average_loss,probs,labels=test_one_epoch(test_loader,model,device,test_logger)
        # thres=get_classification_metrics2(labels,probs)
        metrics=get_classification_metrics(labels,probs,thres=stopper.thres)
        test_logger.update_metrics([getattr(metrics,i) for i in names], names)
        tmp_out="".join([f"TEST {i}:{getattr(metrics,i):.3f} | " for i in names])
        print(f"Epoch FINAL: "+tmp_out)
        
        with open(scores_file_path,"a",encoding="utf-8") as f:
            f.write(", ".join([f"{getattr(metrics,i):.4f}" for i in names])+"\n")
        scoresss.append([getattr(metrics,i) for i in names])


    scoresss=np.array(scoresss)
    mean=np.mean(scoresss,axis=0)
    std=np.std(scoresss,axis=0)
    print(dir_path)
    with open(scores_file_path,"a",encoding="utf-8") as f:
        for i,j,k in zip(names,mean,std):
            f.write(f"{i}={j:.5f}±{k:.5f} \n")
    print(0)