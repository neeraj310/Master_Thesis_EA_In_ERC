# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 13:39:03 2021

@author: neera
"""

import os
import argparse
import math
import time
import random


import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc

import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning import seed_everything

import Utils
import EmoTrain
def get_args():
    """
        returns the Parser args
    """
    root_dir = os.getcwd()  
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--projection_size', type=int, default=100)
    parser.add_argument('--n_layers', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--bert_lr', type=float, default=1e-5)
    parser.add_argument('--layerwise_decay', default=0.95, type=float,  
                    help='layerwise decay factor for the learning rate of the pretrained DistilBert')
    
    #parser.add_argument('--train_file', default=os.path.join(root_dir, 'data/clean_train.txt'), type=str)
    #parser.add_argument('--val_file', default=os.path.join(root_dir, 'data/clean_val.txt'), type=str)
    #parser.add_argument('--test_file', default=os.path.join(root_dir, 'data/clean_test.txt'), type=str)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--epochs', type=int, default=1,
                        help='number of total epochs to run')
    parser.add_argument('--max_seq_len', type=int, default=50   )
   
    parser.add_argument('--alpha', type=float, default=0.95)
    parser.add_argument('--seed', type=int, default=123,
                       help='seed for initializing training')
    parser.add_argument('--device', type=int, default=0)

    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--gpu', type=str, default='0',		# Spcify the GPU for training
	                    help='gpu: default 0')
    parser.add_argument('--emoset', type=str, default = 'emorynlp',
                        help = 'Emotion Training Set Name')
    parser.add_argument('--speaker_embedding',  action="store_true",
                        help = 'Enable Speaker Embedding')
    #args = parser.parse_args(args=[])
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    print(args, '\n')
    args.emoset = args.emoset.lower()
    assert args.emoset  in ['emorynlp', 'emotionpush', 'friends','semeval']
    if args.emoset == 'semeval':
        args.speaker_embedding = False 
    
    if not args.emoset == 'semeval':
        args.batch_size = 1
     
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    seed_everything(42, workers=True)
    
    device = torch.device("cuda:{}".format(int(args.gpu)) if torch.cuda.is_available() else "cpu")
    args.device = device
    print('Args.device = {}'.format(args.device))
    sns.set(style='whitegrid', palette='muted', font_scale=1.2)
    HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]
    sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))
    rcParams['figure.figsize'] = 12, 8
    
    print('Creating train/eval dfs')
    (df_train, df_val, df_test)  = Utils.load_df(args)   
    print('Finished Creating train/eval dfs')

    if args.emoset == 'emorynlp':
        emo_dict= {'neutral': 0, 'sad': 1, 'mad':2, 'joyful':3, 'peaceful':4,'powerful':5, 'scared':6}
        focus_dict = ['neutral', 'sad', 'mad', 'joyful', 'peaceful', 'powerful', 'scared']
    elif args.emoset == 'friends':
        emo_dict = {'neutral': 0, 'sadness': 1, 'anger':2, 'joy':3, 'non-neutral':4,'surprise':5, 'fear':6, 'disgust':7}
        focus_dict = ['neutral', 'sadness', 'anger', 'joy']
    elif args.emoset == 'emotionpush':
        emo_dict = {'neutral': 0, 'sadness': 1, 'anger':2, 'joy':3, 'non-neutral':4,'surprise':5, 'fear':6, 'disgust':7}
        focus_dict = ['neutral', 'sadness', 'anger', 'joy']
    elif args.emoset == 'semeval':
        emo_dict = {'others': 0, 'sad': 1, 'angry':2, 'happy':3}
        focus_dict = ['sad', 'angry', 'happy']

        print('Emotion Mapping {}'.format(emo_dict))
        print('Focus Emotion {}'.format(focus_dict))
        emo_count = {}
        for emo in emo_dict.keys():
                emo_count[emo] =df_train['label'].value_counts()[emo]
        print('Emotion Label Distribution in Training Set {}'.format(emo_count))
        
        emo_count = {}
        for emo in emo_dict.keys():
                emo_count[emo] =df_val['label'].value_counts()[emo]
        print('Emotion Label Distribution in Validation Set {}'.format(emo_count))
        emo_count = {}
        for emo in emo_dict.keys():
            emo_count[emo] =df_test['label'].value_counts()[emo]
        print('Emotion Label Distribution in Test Set {}'.format(emo_count))

    model = EmoTrain.EmotionModel(emo_dict, focus_dict, args, df_train, df_val, df_test)
    model.cuda(args.device)

    early_stop_callback = pl.callbacks.EarlyStopping(monitor='valid_loss', min_delta=0.0005, patience=3,
                                        verbose=True, mode='min')

    checkpoint_callback = pl.callbacks.ModelCheckpoint( monitor='valid_loss' , mode = 'min', save_top_k = 4) 

  
    gpu_list = [int(args.gpu)]  
    trainer = pl.Trainer(default_root_dir=os.getcwd(),
                    gpus=(gpu_list if torch.cuda.is_available() else 0),
                    max_epochs= args.epochs,
                    fast_dev_run=False,
                    deterministic=True,
                    callbacks = [early_stop_callback,checkpoint_callback] ,
                    )
    trainer.fit(model)
    checkpoint_path = './lightning_logs/version_0/checkpoints/'
    checkpoints = os.listdir(checkpoint_path)

    for checkpoint in checkpoints:
        print(checkpoint)
        trainer.test(ckpt_path = checkpoint_path + checkpoint)


if __name__ == '__main__':
	main()