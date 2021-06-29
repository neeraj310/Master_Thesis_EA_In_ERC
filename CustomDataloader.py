# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 14:33:50 2021

@author: neera
"""

from transformers import DistilBertTokenizer
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import OneHotEncoder
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, df, max_number_of_speakers_in_dialogue, emo_dict, args):
        self.data = df
     
        self.max_number_of_speakers_in_dialogue = max_number_of_speakers_in_dialogue
        self.emo_dict = emo_dict
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.args = args
        
    def __len__(self):
        if not self.args.emoset == 'semeval':
            return len( self.data['dialogue_id'].unique())
        else:
            return self.data.shape[0]
    
    def get_labels(self, df):
        """
        returns the labels according to the emotion dictionary
        """
        return torch.tensor([self.emo_dict[label] for label in df['label'].values])
    
    def transform_data(self, df, max_seq_len):
        """
        returns the padded input ids and attention masks according to the DistilBert tokenizer
        """
       
        
        def tokenize_fct(turn):
            return self.tokenizer.encode(turn, truncation = True, add_special_tokens=True, max_length=max_seq_len)
        
        if not self.args.emoset == 'semeval':
            tokenized = df[['utterance']].applymap(tokenize_fct)
            padded = torch.tensor([[ids + [0]*(max_seq_len-len(ids)) for ids in idx] for idx in tokenized.values])
            attention_mask = torch.where(padded != 0, torch.ones_like(padded), torch.zeros_like(padded))
        else:
             tokenized = df[['turn1','turn2','turn3']].applymap(tokenize_fct)
             padded = torch.tensor([[ids + [0]*(max_seq_len-len(ids)) for ids in idx] for idx in tokenized.values])
             attention_mask = torch.where(padded != 0, torch.ones_like(padded), torch.zeros_like(padded))
        return padded, attention_mask, tokenized

    
    def __getitem__(self, idx):
        
        if not self.args.emoset == 'semeval':
            assert idx  in list(range (len( self.data['dialogue_id'].unique())))
            df = self.data.loc[self.data.dialogue_id ==idx]
        else:
           df = self.data.loc[self.data.index ==idx]
        padded, attention_mask, tokenizer = self.transform_data(df, self.args.max_seq_len)
        labels = self.get_labels(df)
        if not self.args.emoset == 'semeval':
            padded = padded.squeeze(dim=1)
            attention_mask = attention_mask.squeeze(dim=1)
        else:
            padded = padded.squeeze(dim=0)
            attention_mask = attention_mask.squeeze(dim=0)
        padded = padded.cuda(self.args.device)
        attention_mask = attention_mask.cuda(self.args.device)
        labels = labels.cuda(self.args.device)

        if self.args.speaker_embedding:
             cat = OneHotEncoder()
             speakers_array = np.array(df.speaker).reshape(-1, 1)
             number_of_speakers_in_dialogue = np.unique(speakers_array).shape[0]
             spkr_emd = cat.fit_transform(speakers_array).toarray()
             spkr_emd = np.pad(spkr_emd, (0, (self.max_number_of_speakers_in_dialogue - number_of_speakers_in_dialogue)))[0:spkr_emd.shape[0], :]
        else:
            spkr_emd = 0
        
        return padded, attention_mask, spkr_emd, labels


