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
import CustomDataloader
from torch.utils.data import DataLoader

from pytorch_lightning import seed_everything

import Utils
import EmoTrain
import Preprocess

import shutil
def get_args():
    """
        returns the Parser args
    """
    root_dir = os.getcwd()  
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--projection_size', type=int, default=200)
    parser.add_argument('--n_layers', type=int, default=2)
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
    parser.add_argument('--epochs', type=int, default=2,
                        help='number of total epochs to run')
    parser.add_argument('--max_seq_len', type=int, default=35   )
   
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
    parser.add_argument('--random',  action="store_true",
                        help = 'Enable non deterministic seed')
    args = parser.parse_args()
    return args

def main():
    if (os.path.isdir('lightning_logs')):
        shutil.rmtree('lightning_logs')
    args = get_args()
    print(args, '\n')
    args.emoset = args.emoset.lower()
    assert args.emoset  in ['emorynlp', 'emotionpush', 'friends','semeval']
    if args.emoset == 'semeval':
        args.speaker_embedding = False 
    
    if not args.emoset == 'semeval':
        args.batch_size = 1
     
    if not args.random:
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
        
    if 0:
       
        df_train = Preprocess.text_preprocess(df_train)
        df_val = Preprocess.text_preprocess(df_val)
        df_test = Preprocess.text_preprocess(df_test)

    model = EmoTrain.EmotionModel(emo_dict, focus_dict, args, df_train, df_val, df_test, verbosity =1)
    model = model.cuda(args.device)
    trainer = EmoTrain.train_model(model)
    EmoTrain.test_model(trainer, model)
    
    
    print('\n Testing with ni cintext')
    model.context_classifier_model.context_experiment_flag= True
    df_shuffled = df_test.copy(deep = True)
    df_shuffled['dialogue_id'] = df_shuffled.index
    dataset = CustomDataloader.CustomDataset( df_shuffled, model.max_number_of_speakers_in_dialogue, model.emo_dict, model.encoder, model.args)
    
    test_loader = DataLoader(dataset, 5, shuffle=False, num_workers=0)
    trainer.test(test_dataloaders =  test_loader)
    
    test_loader = DataLoader(dataset, 10, shuffle=False, num_workers=0)
    trainer.test(test_dataloaders =  test_loader)
    
    
    test_loader = DataLoader(dataset, 1, shuffle=False, num_workers=0)
    trainer.test(test_dataloaders =  test_loader)
    
    '''
    
    df_train_shuffled = Utils.shuffle_dataframe(df_train)
    df_val_shuffled = Utils.shuffle_dataframe(df_val)
    df_test_shuffled = Utils.shuffle_dataframe(df_test)
    
    model_shuffle_0 = EmoTrain.EmotionModel(emo_dict, focus_dict, args, df_train_shuffled, df_val_shuffled, df_test)
    model_shuffle_0.cuda(args.device)
    trainer_shuffle_0 = EmoTrain.train_model(model_shuffle_0)
    EmoTrain.test_model(trainer_shuffle_0, model_shuffle_0)
    
    
    model_shuffle_1 = EmoTrain.EmotionModel(emo_dict, focus_dict, args, df_train, df_val, df_test_shuffled)
    model_shuffle_1.cuda(args.device)
    trainer_shuffle_1 = EmoTrain.train_model(model_shuffle_1)
    EmoTrain.test_model(trainer_shuffle_1, model_shuffle_1)
    
    model_shuffle_2 = EmoTrain.EmotionModel(emo_dict, focus_dict, args, df_train_shuffled, df_val_shuffled, df_test_shuffled)
    model_shuffle_2.cuda(args.device)
    trainer_shuffle_2 = EmoTrain.train_model(model_shuffle_2)
    EmoTrain.test_model(trainer_shuffle_2, model_shuffle_2)
    
    '''
   


if __name__ == '__main__':
	main()