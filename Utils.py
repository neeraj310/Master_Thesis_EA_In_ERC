	# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 13:52:02 2021

@author: neera
"""
import json
import pickle
import torch
import os
import math
import random
import numpy as np

import time
import pandas as pd


def create_df_emorynlp(data, column_list):
    df = pd.DataFrame(columns = column_list)
    dialogue_idx = 0
    episode_idx = 0
    for episode in data['episodes']:
        scenes = episode['scenes']
        for i in range(len(scenes)):
            for j in range(len(scenes[i]['utterances'])):
                df2 = pd.DataFrame([[dialogue_idx, scenes[i]['utterances'][j]['transcript'] , scenes[i]['utterances'][j]['speakers'][0], len(scenes[i]['utterances'][j]['transcript'].split()), scenes[i]['utterances'][j]['emotion'].lower() ]], columns = column_list)
                df = df.append(df2, ignore_index=True)
            dialogue_idx += 1
        episode_idx = episode_idx+1
        if (episode_idx == 5000 ):  # Load less data for test run
            break
        
    print(episode_idx)
    return df


def create_df_friends(data, column_list):
    df = pd.DataFrame(columns = column_list)
    dialogue_idx = 0
    for dialog in data:
        for utter in dialog:
            df2 = pd.DataFrame([[dialogue_idx, utter['utterance'], utter['speaker'], len(utter['utterance'].split()), utter['emotion'].lower() ]], columns = column_list)
            df = df.append(df2, ignore_index=True)
        dialogue_idx += 1
    
        if (dialogue_idx == 5000):  # Load less data for test run
            break
        
    return df
    
def create_df(data, emoset):
    column_list = list(('dialogue_id', 'utterance', 'speaker','utterance_len', 'label'))
    if emoset == 'emorynlp':
        df=  create_df_emorynlp(data, column_list)
    elif emoset == 'friends' or emoset == 'emotionpush':
        df = create_df_friends(data, column_list)
   
    
    return df

def load_df(args):
    
    if args.emoset == 'emorynlp':
        print('Creating Training/Val/Test Dataframesfor EmoryNLP Dataset')
        train_path = '../data/EmoryNLP/json/emotion-detection-trn.json'
        val_path = '../data/EmoryNLP/json/emotion-detection-dev.json'
        test_path = '../data/EmoryNLP/json/emotion-detection-tst.json'
    elif  args.emoset == 'emotionpush':
        print('Creating Training/Val/Test Dataframesfor EmotionPush Dataset')
        train_path = '../data/Emotionpush/emotionpush.train.json'
        val_path =   '../data/Emotionpush/emotionpush.dev.json'
        test_path =  '../data/Emotionpush/emotionpush.test.json'
    elif  args.emoset == 'friends': 
        print('Creating Training/Val/Test Dataframesfor Friends Dataset')
        train_path = '../data/Friends/friends.train.json'
        val_path = '../data/Friends/friends.dev.json'
        test_path = '../data/Friends/friends.test.json'
    else:
        print('Creating Training/Val/Test Dataframesfor Semeval Dataset')
        train_path = '../data/clean_train.txt'
        val_path = '../data/clean_val.txt'
        test_path = '../data/clean_test.txt'

    if not args.emoset == 'semeval':
        with open(train_path, encoding='utf-8') as data_file:
            train_json = json.loads(data_file.read())
        
        with open(val_path, encoding='utf-8') as data_file:
            val_json = json.loads(data_file.read())
        
        with open(test_path, encoding='utf-8') as data_file:
            test_json = json.loads(data_file.read())
        
        df_train = create_df(train_json, args.emoset)
        df_val = create_df(val_json,  args.emoset)       
        df_test = create_df(test_json,  args.emoset) 
    else:
        df_train = pd.read_csv(train_path, delimiter='\t', index_col='id')
        df_val =  pd.read_csv(val_path, delimiter='\t', index_col='id')    
        df_test = pd.read_csv(test_path, delimiter='\t', index_col='id')
        df_train = df_train.iloc[0:500, :]
        df_val = df_val.iloc[0:160, :]
        df_test = df_test.iloc[0:160, :]
        
    return (df_train, df_val, df_test)
    