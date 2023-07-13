import numpy as np
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.colors as mcolors
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix)
from sklearn.metrics import (precision_score, recall_score, f1_score)
from sklearn.metrics import roc_curve, auc, roc_auc_score
from mlxtend.plotting import plot_confusion_matrix
from tqdm import tqdm
import argparse
import time
import os
import warnings

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from pytorch_lightning import seed_everything
import transformers
from transformers import BertTokenizer, DistilBertTokenizer, get_linear_schedule_with_warmup


from dataloader import TwitterDataset
from model import *
from utils import *



def read_options():
    # construct the argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--exp_name', type=str, default='xxxx',
        help='experiment setting name'
    )
    parser.add_argument(
        '--data_path', type=str, default='./dataset',
        help='path for dataset'
    )
    parser.add_argument(
         '--epochs', type=int, default=5, #5, 10
        help='Number of epochs to train our network for'
    )
    parser.add_argument(
        '--loss_fn', type=str, default='cross_entropy',
        help='loss function'
    )
    parser.add_argument(
        '--model_name', type=str, default='DistilBERT',
        help='model name'
    )
    parser.add_argument(
        '--max_len', type=int, default=128, #64
        help='max length for BERT-based model'
    )
    parser.add_argument(
        '--weight_loss', action='store_true', default=False,
        help='Whether to use weighted loss'
    )
    parser.add_argument(
        '--gamma', type=float, default=1e-4,
        help='initial gamma value'
    )
    parser.add_argument(
        '--pretrained', action='store_true', default=False,
        help='Whether to use pretrained weights or not'
    )
    parser.add_argument(
        '--lr', type=float, default=1e-5,
        help='Learning rate for training the model'
    )
    parser.add_argument(
        '--batch_size', type=int, default=64, #16
        help='batch size for training'
    )
    parser.add_argument(
        '--num_classes', type=int, default=1,
        help='number of label classes'
    )
    parser.add_argument(
        '--num_workers', type=int, default=4,
        help='parellel num workers'
    )
    parser.add_argument(
        '--seed', type=int, default=1000,
        help='seed for experiment'
    )
    parser.add_argument(
        '--mislabel_rate', type=float, default=0,
        help='mislabel rate for training set'
    )
    try:
        parsed = vars(parser.parse_args())
    except IOError as msg:
        parser.error(str(msg))
    
    return parsed




class TwitterDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        text = self.df.text[item]
        label = self.df.label[item]

        encoding = self.tokenizer.encode_plus(
            text, #w/o .item()
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True,
            #padding="longest",    
            pad_to_max_length=True,
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors="pt",
        )

        return {
            "text": text,
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.float),
        }
'''prediction by thrs'''
def pred_thrs(results, lbls):
    
    step = 0.1
    thrs_list = np.arange(0.1,1,step)
    test_columns = ['threshold', 'precision', 'recall', 'f1-score', 'accuracy']
    score_matrix = np.zeros((len(thrs_list),len(test_columns)), dtype=object)
    
    best_f1 = 0
    best_pred = np.zeros(len(lbls))
    #for idx in range(1, len(thrs_list)):
    for idx, threshold in enumerate(thrs_list):
        print('-'*50)
        print('threshold:', threshold.round(1))
        
        #for t, scores in enumerate(results):
        test = np.copy(results) #copy for not covered
        test = np.where(test>threshold, 1, 0)
        #print(test, labels)
        '''compute metrics'''
        prec = precision_score(lbls, test).round(4)
        recall = recall_score(lbls, test).round(4)
        f1 = f1_score(lbls, test).round(4)
        acc = accuracy_score(lbls, test).round(4)
        '''update best predictions'''
        #if f1 > best_f1:
        if threshold == 0.5:
            best_pred = test
        
        print(f'precision:{prec}/recall:{recall}/f1-score:{f1}/acc:{acc}')
        
        score_matrix[idx] = [threshold.round(1), prec, recall, f1, acc]
    score_result = pd.DataFrame(score_matrix, columns=test_columns)   
    return score_result, best_pred


#main function
def main():
    
    warnings.filterwarnings('ignore')
    
    '''read parsed'''
    args = read_options()
    exp_name = args['exp_name']
    n_epochs = args['epochs']
    mislabel_rate = args['mislabel_rate']
    '''loss function'''
    criterion = args['loss_fn']
    
    '''set seed'''
    seed_everything(args['seed'])
    
    '''set device'''
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Computation device: {device}")
    
    '''model and tokenizer'''
    if args['model_name'] == 'DistilBERT':
        model = DistilBERT_Sent(args['num_classes']).to(device)
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    if args['model_name'] == 'BERT':
        model = BERT_Sent(args['num_classes']).to(device)
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
    '''load best model'''
    model.load_state_dict(torch.load(f'./outputs/model_ckpt/{exp_name}/model_best.pth')['model_state_dict'])
    
    '''dataset'''
    data_path = args['data_path']
    test_df = pd.read_csv(os.path.join(data_path,'test.csv'), encoding='ISO-8859-1')
    test_ds = TwitterDataset(test_df, tokenizer, args['max_len'])
    test_loader = DataLoader(test_ds, args['batch_size'], num_workers=args['num_workers'])
    
    
    '''model eval'''
    model.eval()
    labels_full, proba_full, losses = [], [], []
    corrects = 0
    with torch.no_grad():
        for i, d in enumerate(tqdm(test_loader)):
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            labels = d["labels"].to(device)
        
            outputs = model(input_ids=input_ids, attention_mask=attention_mask).view(-1)
            temp_outs - torch.sigmoid(outputs)
            preds = torch.where(temp_outs>0.5,1,0)
            
            '''loss function'''
            if criterion == 'gamma_logit_loss':
                loss, _ = gamma_logit_loss(outputs, labels)
            else:
                loss = nn.BCEWithLogitsLoss()(outputs, labels)
                #loss = nn.BCELoss()(outputs, labels)

            corrects += torch.sum(preds == labels).item()
            labels_full += labels.data.cpu().numpy().tolist()
            proba_full += temp_outs.data.cpu().numpy().tolist()
            losses.append(loss.detach().cpu().numpy())
            
    eval_acc = round(corrects/len(test_ds), 4)
    eval_loss = round(np.mean(losses), 4)
    print(f'testing acc: {eval_acc}| testing loss: {eval_loss}')
    #eval_files = list(range(1,len(test_ds)+1))
    thrs_df, prediction = pred_thrs(proba_full, labels_full)
    out_df = pd.DataFrame(np.array([prediction, labels_full], dtype=object).T, columns=['prediction', 'label'])
    print('EVALUATION COMPLETE')
    print('-'*50)
    
    save_dir = f'./outputs/logs/{exp_name}/'
    os.makedirs(save_dir, exist_ok=True)
    #output to csv
    thrs_df.to_csv(os.path.join(save_dir, 'thrs_report.csv'), index=False)
    out_df.to_csv(os.path.join(save_dir, 'pred_report.csv'), index=False)
    
    
    
    #classification report
    print('*********************************************************')
    print(classification_report(labels_full, prediction))
    
    
    #confusion matrix
    conf_matrix = confusion_matrix(prediction, labels_full)
    fig1, ax = plot_confusion_matrix(conf_mat=conf_matrix, figsize=(6, 6))
    plt.xlabel('Actuals ', fontsize=18)
    plt.ylabel('Predictions', fontsize=18)
    plt.title('Confusion Matrix', fontsize=18)
    plt.savefig(os.path.join(save_dir, 'conf_mtrx.png'))
    
    #ROC and AUC score
    #predicted probability
    fpr, tpr, _ = roc_curve(labels_full,  proba_full)
    auc = np.round(roc_auc_score(labels_full, proba_full),4)
    fig2 = plt.figure(figsize = (6,6))
    plt.plot(fpr,tpr,label="ROC, with AUC="+str(auc))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc=4)
    plt.savefig(os.path.join(save_dir, 'roc_auc.png'))
    
if __name__ == '__main__':
    main()