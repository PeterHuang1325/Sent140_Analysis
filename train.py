import torch
from torch import nn
from tqdm import tqdm
import numpy as np
from collections import defaultdict, Counter
import os
from utils import *

def train(model, data_loader, criterion, optimizer, scheduler, gamma, mislabel_rate, n_samples, device):
    model = model.train()

    losses = []
    corrects = 0
    gamma_full = []
    pos = 0
    for i, d in enumerate(tqdm(data_loader)):
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        labels = d["labels"].to(device)
        '''add mislabel'''
        if mislabel_rate > 0:
            mis_idx = round(mislabel_rate*len(labels))
            labels[:mis_idx] = 1 - labels[:mis_idx]
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask).view(-1)
        #print(outputs)
        preds = torch.where(torch.sigmoid(outputs)>0.5,1,0)
        pos += torch.sum(preds)
        #print(pos)
        '''loss function'''
        if criterion == 'gamma_logit_loss':
            loss, gam_update = gamma_logit_loss(outputs, labels, gamma)
            gamma_full.append(gam_update)
        else:
            loss = nn.BCEWithLogitsLoss()(outputs, labels)
            #loss = nn.BCELoss()(outputs, labels)
        #print(loss)

        loss.backward()
        corrects += torch.sum(preds == labels).item()
        #print(corrects)
        losses.append(loss.detach().cpu().numpy())
        
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
    gamma = np.median(gamma_full)
    #gamma = gam_update
    train_acc = round(corrects/n_samples,3)
    train_loss = round(np.mean(losses),3)
    return train_acc, train_loss, gamma


def evaluation(model, data_loader, criterion, gamma, mislabel_rate, n_samples, device):
    model = model.eval()

    losses = []
    corrects = 0

    with torch.no_grad():
        for d in tqdm(data_loader):
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            labels = d["labels"].to(device)
            '''add mislabel'''
            if mislabel_rate > 0:
                mis_idx = round(mislabel_rate*len(labels))
                labels[:mis_idx] = 1 - labels[:mis_idx]
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask).view(-1)
            preds = torch.where(torch.sigmoid(outputs)>0.5,1,0)
            
            '''loss function'''
            if criterion == 'gamma_logit_loss':
                loss, _ = gamma_logit_loss(outputs, labels, gamma)
            else:
                loss = nn.BCEWithLogitsLoss()(outputs, labels)
                #loss = nn.BCELoss()(outputs, labels)

            corrects += torch.sum(preds == labels).item()
            losses.append(loss.detach().cpu().numpy())
    eval_acc = round(corrects/n_samples, 3)
    eval_loss = round(np.mean(losses), 3)
    return eval_acc, eval_loss


def train_model(exp_name, model, train_loader, val_loader, criterion, optimizer, scheduler, n_epochs, gamma, mislabel_rate, n_train, n_val, save_dir, device):
    history = defaultdict(list)
    best_acc = 0
    '''check save dir'''
    os.makedirs(save_dir, exist_ok=True) 
    '''training'''
    for epoch in range(n_epochs):
        print('-' * 50)
        print(f'Epoch {epoch + 1}/{n_epochs}')

        '''training'''
        train_acc, train_loss, gamma = train(model, train_loader, criterion, optimizer, scheduler, gamma, mislabel_rate, n_train, device)
        
        '''validation'''
        val_acc, val_loss = evaluation(model, val_loader, criterion, gamma, mislabel_rate, n_val, device)
        
        '''write results''' 
        with open(os.path.join(save_dir, 'history.txt'),'a') as f:
            '''print results'''
            print(f"[epoch {epoch+1}]: train_loss = {train_loss:.4f}, train_acc = {train_acc:.4f}, val_loss = {val_loss:.4f}, val_acc = {val_acc:.4f}, gamma={gamma:.4f}")
            '''write to file'''
            print(f"[epoch {epoch+1}]: train_loss = {train_loss:.4f}, train_acc = {train_acc:.4f}, val_loss = {val_loss:.4f}, val_acc = {val_acc:.4f}, gamma={gamma:.4f}", file=f)
        
        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss)
        history['gamma'].append(gamma)
        
        if val_acc > best_acc:
            torch.save({'model_state_dict': model.state_dict()}, f"{save_dir}/model_best.pth")
            best_acc = val_acc
    return history