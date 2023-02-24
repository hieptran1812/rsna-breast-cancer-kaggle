import os
import numpy as np
import time
import random
import pandas as pd
from pathlib import Path
from logult import setup_log
from datetime import datetime
from tqdm import tqdm

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from timm.models.efficientnet import *
from sklearn.model_selection import StratifiedGroupKFold

import albumentations as A
from sklearn.metrics import roc_auc_score

from dataset import RsnaDataset
from model import RsnaModel
from utils import load_config


# Seed all random number generators
def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def batch_to_device(batch, device):
    batch_dict = {key: batch[key].to(device) for key in batch}
    return batch_dict


def pfbeta(labels, predictions, beta=1.):
    #official implementation
    y_true_count = 0
    ctp = 0
    cfp = 0

    for idx in range(len(labels)):
        prediction = min(max(predictions[idx], 0), 1)
        if (labels[idx]):
            y_true_count += 1
            ctp += prediction
#             cfp += 1 - prediction #bugfix
        else:
            cfp += prediction

    beta_squared = beta * beta
    c_precision = ctp / (ctp + cfp)
    c_recall = ctp / y_true_count
    if (c_precision > 0 and c_recall > 0):
        result = (1 + beta_squared) * (c_precision * c_recall) / (beta_squared * c_precision + c_recall)
        return result
    else:
        return 0

    
def optimal_f1(labels, predictions):
    thres = np.linspace(0, 1, 101)
    f1s = [pfbeta(labels, predictions > thr) for thr in thres]
    idx = np.argmax(f1s)
    return f1s[idx], thres[idx]
    

def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or np.any([v in name.lower()  for v in skip_list]):
            # print(name, 'no decay')
            no_decay.append(param)
        else:
            # print(name, 'decay')
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]
     

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='../config/default.yaml', help='config path')
    parser.add_argument('--device', type=str, default='cuda:0', help='choose device manager')
    parser.add_argument('--log', type=str, default='info.log', help='logname file')
    opt = parser.parse_args()
    return opt

    
if __name__ == '__main__':
    cfg = load_config('config/default.yaml')
    seed_everything(seed=cfg.seed)
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    # logger
    cur_time = str(datetime.now())
    cur_time = cur_time.replace(' ', '_')
    cur_time = cur_time.split('.')[0]
    log_name = '{}_{}'.format(cur_time, cfg.backbone)
    Path('saved/weights/{}'.format(log_name)).mkdir(parents=True, exist_ok=True)
    Path('saved/log/{}'.format(log_name)).mkdir(parents=True, exist_ok=True)
    logger = setup_log('saved/log/{}'.format(log_name), 'info.log')
    # dataset
    df_train = pd.read_csv('data/train.csv')
    split = StratifiedGroupKFold(cfg.folds)
    for k, (_, test_idx) in enumerate(split.split(df_train, df_train.cancer, groups=df_train.patient_id)):
        df_train.loc[test_idx, 'split'] = k
    df_train.split = df_train.split.astype(int)
    
    aug = {
        "train": A.Compose([
                A.Resize(cfg.image_size, cfg.image_size),
                # A.HorizontalFlip(p=0.1),
            ]),
        "val": A.Compose([
            A.Resize(cfg.image_size, cfg.image_size),
        ])}
    
    for fold in range(2, cfg.folds):
        train_dataset = RsnaDataset(df_train.query('split != @fold'), cfg, aug['train'])
        train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)
        val_dataset = RsnaDataset(df_train.query('split == @fold'), cfg, aug['val'])
        val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)
        logger.info('Load dataset successful with {} sample train, {} sample valid'.format(len(train_dataset), len(val_dataset)))
        model = RsnaModel(cfg).to(device)
        logger.info('Load model successful!')
        # optimizer
        # optim = torch.optim.AdamW(add_weight_decay(model, weight_decay=cfg.adamw_decay, skip_list=['bias']), lr=cfg.lr, betas=(0.9, 0.999), weight_decay=cfg.adamw_decay)
        # scheduler = torch.optim.lr_scheduler.OneCycleLR(optim, max_lr=cfg.lr, epochs=cfg.epochs, steps_per_epoch=len(train_dataset))
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-6, last_epoch=-1)
        scaler = GradScaler()
        best_eval_score = 0 
        running_loss = None
        for epoch in tqdm(range(cfg.epochs)):
            model.train()
            loss = None
            
            for step, batch in enumerate(train_loader):
                batch = batch_to_device(batch, device)
                
                optimizer.zero_grad()
                with autocast():
                    loss = model(batch)['loss']
                    scaler.scale(loss).backward()
                    if running_loss is None:
                        running_loss = loss.item()
                    else:
                        running_loss = running_loss * 0.9 + loss.item() * 0.1
                    # running_loss = loss.item()
                    if step % 200 == 0:
                        logger.info(f'From epoch {epoch} after {step} iter, loss is {running_loss}')
                    if (step + 1 == len(train_loader)) or ((step+1) % cfg.accum_iter == 0):
                        
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                        if scheduler is not None:
                            scheduler.step()
                    if step + 1 == len(train_loader):                                                  
                        logger.info('Epoch {} loss: {}'.format(epoch, running_loss))
            if scheduler is not None:
                scheduler.step()
            
            with torch.no_grad():
                model.eval()
                image_preds_all = []
                image_targets_all = []
                t = time.time()
                for step, batch in enumerate(val_loader):
                    batch = batch_to_device(batch, device)
                    label = batch['target']
                    preds = model(batch)['logits']
                    preds = preds.sigmoid().float().detach().cpu().numpy()
                    image_preds_all += [preds]
                    # image_preds_all += [torch.argmax(preds, 1).detach().cpu().numpy()]
                    image_targets_all += [label.detach().cpu().numpy()]
                image_preds_all = np.concatenate(image_preds_all).reshape(-1, )
                image_targets_all = np.concatenate(image_targets_all).reshape(-1, )
                
                # image_targets_all = image_targets_all.reshape(-1)
                # print('image_preds_all', image_preds_all)
                # print('image targets_all', image_targets_all)
                # score = pfbeta(image_targets_all, image_preds_all, 1)
                patient_id = df_train.query('split == @fold')['patient_id'].values
                laterality = df_train.query('split == @fold')['laterality'].values
                prediction_id = [f'{i}_{j}' for i, j in zip(patient_id, laterality)]
                pred_df = pd.DataFrame({'prediction_id': prediction_id, 'cancer_raw': image_preds_all})
                sub = pred_df.groupby('prediction_id')[['cancer_raw']].agg('mean')
                # binarize predictions
                # th = np.quantile(sub['cancer_raw'].values, 0.97935)
                # sub['cancer'] = (sub['cancer_raw'].values > th).astype(int)
                
                test_df = df_train.query('split == @fold') 
                test_df['prediction_id'] = test_df.apply(lambda x: f'{x.patient_id}_{x.laterality}', 1)
                test_df_gr = test_df.groupby('prediction_id')[['cancer']].agg('mean')
                test_df_gr = test_df_gr.loc[sub.index]
                y = test_df_gr['cancer'].values
                y_pred = sub['cancer_raw'].values
                score, thresh = optimal_f1(y, y_pred)
                # score, thresh = optimal_f1(image_targets_all, image_preds_all)
                roc_auc = roc_auc_score(image_targets_all, image_preds_all)
                logger.info('Best threshold archieve in {}, and the score is {}'.format(thresh, score))
                logger.info('CV score {} and AUC_ROC score {}'.format(score, roc_auc))
                if scheduler is not None:
                    scheduler.step()
                if score > best_eval_score:
                    # torch.save(model.state_dict(), 'weights/{}_fold{}_best.pth'.format(cfg.backbone, fold))
                    torch.save(model.state_dict(), 'saved/weights/{}/fold{}_best.pth'.format(log_name, fold))
                torch.save(model.state_dict(), 'saved/weights/{}/fold{}_epoch{}_score{:.2f}_thresh{:.2f}'.format(log_name, fold, epoch, score, thresh))
                # torch.save(model.state_dict(), 'weights/{}_fold{}_epoch{}_score{}_thresh{}.pth'.format(cfg.backbone, fold, epoch, score, thresh))
        del model, optimizer, train_loader, val_loader, scheduler, scaler
        # torch.cuda.empty_cache()
       
