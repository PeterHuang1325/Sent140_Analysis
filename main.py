import numpy as np
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.colors as mcolors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
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


from dataloader import TwitterDataset, TwitterLogits
from model import *
from utils import *
from train import train_model


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
        '--gamma', type=float, default=1e-4, #1e-4, 1e-5
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

#main function
def main():
    
    '''read parsed'''
    args = read_options()
    exp_name = args['exp_name']
    n_epochs = args['epochs']
    mislabel_rate = args['mislabel_rate']
    gam_init = args['gamma']
    print(f'mislabel rate={mislabel_rate}')
    '''set seed'''
    seed_everything(args['seed'])
    
    '''read data'''
    data_path = args['data_path']
    train_df = pd.read_csv(os.path.join(data_path,'train.csv'), encoding='ISO-8859-1')
    val_df = pd.read_csv(os.path.join(data_path,'val.csv'), encoding='ISO-8859-1')
    
    '''set device'''
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    '''model and tokenizer'''
    if args['model_name'] == 'DistilBERT':
        model = DistilBERT_Sent(args['num_classes']).to(device)
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    if args['model_name'] == 'BERT':
        model = BERT_Sent(args['num_classes']).to(device)
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
    '''number of parameters'''
    train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total trainable parameters: {train_params}')
    
    '''loss function'''
    criterion = args['loss_fn']
    
    '''Create torch dataset'''
    train_ds = TwitterDataset(train_df, tokenizer, args['max_len'])
    val_ds = TwitterDataset(val_df, tokenizer, args['max_len'])
        
    '''dataloader'''
    train_loader = DataLoader(train_ds, args['batch_size'], num_workers=args['num_workers'])
    val_loader = DataLoader(val_ds, args['batch_size'], num_workers=args['num_workers'])
   

    '''optimizer'''
    optimizer = Adam(model.parameters(), lr=args['lr'], weight_decay=1e-5)
    total_steps = len(train_loader)*n_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    
    '''train model'''
    warnings.filterwarnings('ignore')
    save_dir = f'./outputs/model_ckpt/{exp_name}/'
    history = train_model(exp_name, model, train_loader, val_loader, criterion, optimizer, scheduler, n_epochs, gam_init, mislabel_rate, len(train_ds), len(val_ds), save_dir, device)
    results_df = pd.DataFrame(np.array([history['train_acc'], history['train_loss'], history['val_acc'], history['val_loss'], history['gamma']]).T, columns=['train_acc','train_loss', 'val_acc','val_loss', 'gamma'])
    results_df.to_csv(os.path.join(save_dir,'results.csv'), index=False)
    print('TRAINING COMPLETE')
    print('-'*50)

if __name__ == '__main__':
    main()