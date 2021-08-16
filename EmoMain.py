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
    parser.add_argument('--projection_size', type=int, default=768)
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--bert_lr', type=float, default=1e-5)
    parser.add_argument('--layerwise_decay', default=0.95, type=float,  
                    help='layerwise decay factor for the learning rate of the pretrained DistilBert')
    

    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--epochs', type=int, default=5 ,
                        help='number of total epochs to run')
  
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
    parser.add_argument('--use_distilbert',  action="store_true",
                        help = 'Use DistilBert for German Text Emotion Prediction')
    
    parser.add_argument('--bww',  action="store_true",
                              help = 'Use balancec weight warming for loss function')
    args = parser.parse_args()
    return args

def main():
    # Remove exiting logs
    if (os.path.isdir('lightning_logs')):
        shutil.rmtree('lightning_logs')
    args = get_args()
    print(args, '\n')
    args.emoset = args.emoset.lower()
    assert args.emoset  in ['emorynlp', 'emotionpush', 'friends','semeval', 'friends_german',
                        'emotionpush_german', 'emorynlp_german', 'semeval_german']
    
    # Speaker Embedding is not supported for Semeval Task.
    if args.emoset == 'semeval':
        args.speaker_embedding = False 
    
    if not args.emoset == 'semeval':
        args.batch_size = 1
        
    
    
    '''
    Speaker Embedding is not supported for Semeval Task.
    distilbert-base-german-cased
    dbmdz/bert-base-german-uncased
    '''
    if args.emoset not in ['friends_german', 'emotionpush_german', 'emorynlp_german']:
        args.use_distilbert = False
        
    if args.emoset == 'friends' or args.emoset == 'friends_german':
            args.max_seq_len = 40
            args.balance_weight_warming =False
    elif args.emoset == 'emotionpush' or args.emoset == 'emotionpush_german':
        args.max_seq_len = 35
        args.balance_weight_warming =False
    elif args.emoset == 'emorynlp' or args.emoset == 'emorynlp_german':
        args.max_seq_len = 30
        args.balance_weight_warming =True
    if args.emoset == 'semeval' or args.emoset == 'semeval_german':
        args.max_seq_len = 25
        args.balance_weight_warming =True
        
    
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

    if args.emoset == 'emorynlp' or args.emoset == 'emorynlp_german':
        emo_dict= {'neutral': 0, 'sad': 1, 'mad':2, 'joyful':3, 'peaceful':4,'powerful':5, 'scared':6}
        focus_dict = ['neutral', 'sad', 'mad', 'joyful', 'peaceful', 'powerful', 'scared']
        sentiment_dict = {'neutral': 0, 'sad': 1, 'mad':1, 'joyful':2, 'peaceful':2,'powerful':2, 'scared':1}
    elif args.emoset == 'friends' or args.emoset == 'friends_german':
    
        emo_dict = {'neutral': 0, 'sadness': 1, 'anger':2, 'joy':3, 'non-neutral':4,'surprise':5, 'fear':6, 'disgust':7}
        focus_dict = ['neutral', 'sadness', 'anger', 'joy']
        sentiment_dict = {'neutral': 0, 'sadness': 1, 'anger':1, 'joy':2, 'non-neutral':0, 'surprise':2,'fear':1, 'disgust':1}
    elif args.emoset == 'emotionpush' or args.emoset == 'emotionpush_german':
      
        emo_dict = {'neutral': 0, 'sadness': 1, 'anger':2, 'joy':3, 'non-neutral':4,'surprise':5, 'fear':6, 'disgust':7}
        focus_dict = ['neutral', 'sadness', 'anger', 'joy']
        sentiment_dict = {'neutral': 0, 'sadness': 1, 'anger':1, 'joy':2, 'non-neutral':0, 'surprise':2,'fear':1, 'disgust':1}
    elif args.emoset == 'semeval':
    
        emo_dict = {'others': 0, 'sad': 1, 'angry':2, 'happy':3}
        focus_dict = ['sad', 'angry', 'happy']
        sentiment_dict = {'others': 0, 'sad': 1, 'angry':1, 'happy':2}
    
        
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
    
 
    model = EmoTrain.EmotionModel(emo_dict, focus_dict, args, df_train, df_val, df_test, verbosity =1)
    model = model.cuda(args.device)
    trainer = EmoTrain.train_model(model)
    EmoTrain.test_model(trainer, model)
    
    if args.emoset in ['friends_german', 'emotionpush_german', 'emorynlp_german', 'semeval']:
        return
    
    '''
    To understand the importance of context, we try to classify the correct label after shuffling 
    the utterances with in a dialogue. For example, a dialogue having utterance sequence of
    {u1;u2;u3;u4} is shuffled to {u4;u1;u3;u2}. This shuffling is carried out randomly, resulting in 
    an utterance sequence whose order is different from the original sequence. We implement 3 shuffling strategies
    
    dialogues in train and validation sets are shuffled, dialogues in test set are kept unchanged
    dialogues in train and validation sets are unchanged, dialogues in test set are shuffled
    dialogues in train and validation and test set are shuffled.
    We found that, whenever there is some shuffling in train, validation, or test set, the performance
    decreases a few points. The performance drop is highest when the dialogues in train and validation sets are 
    kept unchanged and dialogues in test set are shuffled
    '''
    
    print('Result Analysis Code')

    model.context_classifier_model.context_experiment_flag= True
    df_shuffled = df_test.copy(deep = True)
    df_shuffled['dialogue_id'] = df_shuffled.index
    
    print('Test Model performance without any context: 1 utterance per conversation')
    dataset = CustomDataloader.CustomDataset( df_shuffled, model.max_number_of_speakers_in_dialogue, model.emo_dict, model.encoder, model.args)
    test_loader = DataLoader(dataset, 1, shuffle=False, num_workers=0)
    trainer.test(test_dataloaders =  test_loader)
    
    
    print('Test Model performance with context restricted to 5 utterances per conversation')
    test_loader = DataLoader(dataset, 5, shuffle=False, num_workers=0)
    trainer.test(test_dataloaders =  test_loader)
    
    print('Test Model performance with context restricted to 10 utterances per conversation')
    test_loader = DataLoader(dataset, 10, shuffle=False, num_workers=0)
    trainer.test(test_dataloaders =  test_loader)
    
    print('Test Model performance with context restricted to single speaker per conversation')
    speaker_list = df_test['speaker'].value_counts()[:6].index.tolist()
    df_without_interspeaker_context = {}

    for speaker in speaker_list:
        df_without_interspeaker_context[speaker] = df_test[(df_test.speaker == speaker)].reset_index(drop=True)
        dialogue_id_list = list((df_without_interspeaker_context[speaker]['dialogue_id'].unique()))
        i = 0
        for dialogue in dialogue_id_list:
            df_without_interspeaker_context[speaker].loc[df_without_interspeaker_context[speaker].dialogue_id== dialogue, 'dialogue_id'] = i 
            i = i+1
            
    avg_f1_score = 0
    avg_val_loss = 0
    model.context_classifier_model.context_experiment_flag= False
    model.verbosity=False
    for speaker in speaker_list:
        dataset = CustomDataloader.CustomDataset( df_without_interspeaker_context[speaker], model.max_number_of_speakers_in_dialogue, model.emo_dict, model.encoder, model.args)
        test_loader = DataLoader(dataset, model.args.batch_size, shuffle=False, num_workers=0)
        score = trainer.test(test_dataloaders =  test_loader)
        print('F1 Score for Speaker {} is {}'.format(speaker, score[0]['macroF1'] ))
        avg_f1_score += score[0]['macroF1']
        avg_val_loss += score[0]['valid_loss']

    print('F1 Score without Intraspeaker Context {}'.format(avg_f1_score/len(speaker_list)))
    print('Validation Loss  without Intraspeaker Context {}'.format(avg_val_loss/len(speaker_list)))
    model.verbosity=True

    
    
    '''
    Classification in shuffled context. 
    We tried following 3 shuffling techniques and measure model performance 
    to analyze the impact of context shuffling. 
    
    a) Dialogues in train and validation sets are shuffled,  dialogues in test set are kept un-changed.
    b) Dialogues in test set are shuffled, whereas dialogues in train and validation sets set arekept unchanged
    c) Dialogues in all three train, validation and test sets are shuffled
    '''
  
    df_train_shuffled = Utils.shuffle_dataframe(df_train)
    df_val_shuffled = Utils.shuffle_dataframe(df_val)
    df_test_shuffled = Utils.shuffle_dataframe(df_test)
    
    print('Test Model performance with shuffled context : Dialogues in train and validation sets are shuffled, dialogues in test set are kept un-changed')
    model_shuffle_0 = EmoTrain.EmotionModel(emo_dict, focus_dict, args, df_train_shuffled, df_val_shuffled, df_test)
    model_shuffle_0.cuda(args.device)
    trainer_shuffle_0 = EmoTrain.train_model(model_shuffle_0)
    EmoTrain.test_model(trainer_shuffle_0, model_shuffle_0)
  
    print('Test Model performance with shuffled context : Dialogues in test set are shuffled, whereas dialogues in train and validation sets set arekept unchanged')
    model_shuffle_1 = EmoTrain.EmotionModel(emo_dict, focus_dict, args, df_train, df_val, df_test_shuffled)
    model_shuffle_1.cuda(args.device)
    trainer_shuffle_1 = EmoTrain.train_model(model_shuffle_1)
    EmoTrain.test_model(trainer_shuffle_1, model_shuffle_1)
    
    print('Test Model performance with shuffled context : Dialogues in all three train, validation and test sets are shuffled')
    model_shuffle_2 = EmoTrain.EmotionModel(emo_dict, focus_dict, args, df_train_shuffled, df_val_shuffled, df_test_shuffled)
    model_shuffle_2.cuda(args.device)
    trainer_shuffle_2 = EmoTrain.train_model(model_shuffle_2)
    EmoTrain.test_model(trainer_shuffle_2, model_shuffle_2)

   


if __name__ == '__main__':
	main()