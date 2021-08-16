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
     
        

    return df


def create_df_friends(data, column_list):
    df = pd.DataFrame(columns = column_list)
    dialogue_idx = 0
    for dialog in data:
        for utter in dialog:
            df2 = pd.DataFrame([[dialogue_idx, utter['utterance'], utter['speaker'], len(utter['utterance'].split()), utter['emotion'].lower() ]], columns = column_list)
            df = df.append(df2, ignore_index=True)
        dialogue_idx += 1
    
      
        
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
        print('Creating Training/Val/Test Dataframes for EmoryNLP Dataset')
        train_path = '../../data/EmoryNLP/English/emotion-detection-trn.json'
        val_path = '../../data/EmoryNLP/English/emotion-detection-dev.json'
        test_path = '../../data/EmoryNLP/English/emotion-detection-tst.json'
    elif args.emoset == 'emorynlp_german':
        print('Creating Training/Val/Test Dataframes for EmoryNLP German Dataset')
        train_path = '../../data/EmoryNLP/German/emorynlp_train_de.csv'
        val_path = '../../data/EmoryNLP/German/emorynlp_val_de.csv'
        test_path = '../../data/EmoryNLP/German/emorynlp_test_de.csv'

    elif  args.emoset == 'friends': 
        print('Creating Training/Val/Test Dataframes for Friends Dataset')
        train_path = '../../data/Friends/Friends_English/friends.train.json'
        val_path = '../../data/Friends/Friends_English/friends.dev.json'
        test_path = '../../data/Friends/Friends_English/friends.test.json'
    elif args.emoset == 'friends_german':
        print('Creating Training/Val/Test Dataframes for Friends German Dataset')
        train_path = '../../data/Friends/Friends_German/friends_train_de.csv'
        val_path = '../../data/Friends/Friends_German/friends_val_de.csv'
        test_path = '../../data/Friends/Friends_German/friends_test_de.csv'
        
    elif  args.emoset == 'emotionpush':
        print('Creating Training/Val/Test Dataframes for EmotionPush Dataset')
        train_path = '../../data/Emotionpush/English/emotionpush.train.json'
        val_path =   '../../data/Emotionpush/English/emotionpush.dev.json'
        test_path =  '../../data/Emotionpush/English/emotionpush.test.json'
    
    elif args.emoset == 'emotionpush_german':
        print('Creating Training/Val/Test Dataframes for Emotionpush German Dataset')
        train_path = '../../data/Emotionpush/German/emotionpush_train_de.csv'
        val_path =   '../../data/Emotionpush/German/emotionpush_val_de.csv'
        test_path =  '../../data/Emotionpush/German/emotionpush_test_de.csv'
          
    else:
        print('Creating Training/Val/Test Dataframes for Semeval Dataset')
        train_path = '../../data/clean_train.txt'
        val_path = '../../data/clean_val.txt'
        test_path = '../../data/clean_test.txt'

    if args.emoset in ['friends', 'emotionpush', 'emorynlp'] :
        with open(train_path, encoding='utf-8') as data_file:
            train_json = json.loads(data_file.read())
        
        with open(val_path, encoding='utf-8') as data_file:
            val_json = json.loads(data_file.read())
        
     
        with open(test_path, encoding='utf-8') as data_file:
            test_json = json.loads(data_file.read())
        
        df_train = create_df(train_json, args.emoset)
        df_val = create_df(val_json,  args.emoset)       
        df_test = create_df(test_json,  args.emoset) 
    elif args.emoset == 'semeval':
        df_train = pd.read_csv(train_path, delimiter='\t', index_col='id')
        df_val =  pd.read_csv(val_path, delimiter='\t', index_col='id')    
        df_test = pd.read_csv(test_path, delimiter='\t', index_col='id')
        '''
        df_train = df_train.iloc[0:500, :]
        df_val = df_val.iloc[0:160, :]
        df_test = df_test.iloc[0:160, :]
        '''
    else:

        df_train = pd.read_csv(train_path)
        df_val =  pd.read_csv(val_path)    
        df_test = pd.read_csv(test_path)
        df_train = df_train.dropna()
        df_val = df_val.dropna()
        df_test = df_test.dropna()
        col_dict = {'emotion': 'label'}   ## key→old name, value→new name
        df_train.columns = [col_dict.get(x, x) if x in col_dict.keys() else x  for x in df_train.columns]
        df_val.columns = [col_dict.get(x, x) if x in col_dict.keys() else x  for x in df_val.columns]
        df_test.columns = [col_dict.get(x, x) if x in col_dict.keys() else x  for x in df_test.columns]
        #df_train.rename(columns={'emotion':'label'}, inplace=True)
        df_train['utterance_len'] = df_train[['utterance_de_deepl']].applymap(lambda x: len(x.split()))
        df_val['utterance_len'] = df_val[['utterance_de_deepl']].applymap(lambda x: len(x.split()))
        df_test['utterance_len'] = df_test[['utterance_de_deepl']].applymap(lambda x: len(x.split()))
    return (df_train, df_val, df_test)

def shuffle_dataframe(df):
    dialogue_id_list = list((df['dialogue_id'].unique()))
    df_shuffled =  pd.DataFrame(columns = list(df.columns))
    for dialogue in dialogue_id_list:
        df_dialogue = df[(df.dialogue_id ==dialogue)].copy(deep = True)
        df_dialogue = df_dialogue.sample(frac=1)
        df_shuffled = df_shuffled.append(df_dialogue)
    df_shuffled =df_shuffled.reset_index(drop=True)
    return df_shuffled
    