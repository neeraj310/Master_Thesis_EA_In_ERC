# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 14:33:50 2021

@author: neera
"""

from transformers import DistilBertTokenizer
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from transformers import AutoTokenizer

'''
  Create the DataLoader Class to provide a way for the dataset code to be decoupled from  model training code.
  Parameters
    ----------
    df : Pandas Dataframe 
        Pandas dataframe repersenting Train/Val/Test Data
    max_number_of_speakers_in_dialogue : Interger 
    emo_dict : Python Dictionary
    Prvides encoded value of emotion labels. 
    encoder : Sklearn OneHot Encoder Handle

  Returns
    -------
  padded : Tensor
     Tensor containing the input data
  attention : Tensor
     Tensor containing attention mask specifying the valid data in input tensor
  spkr_emd : Tensor
     Tensor containing speaker information in the input data
  label : Tensor
     Tensor containing emotion labels
     
'''



class CustomDataset(Dataset):
    def __init__(self, df, max_number_of_speakers_in_dialogue, emo_dict, encoder, args):
        self.data = df
        self.max_number_of_speakers_in_dialogue = max_number_of_speakers_in_dialogue
        self.emo_dict = emo_dict
        self.args = args
        self.encoder = encoder
        if self.args.emoset == 'friends_german':
            if self.args.use_distilbert: 
                print('Dataset Name {}  Use DistilBert Flag  {} Tokenizer being used {}'.format(args.emoset, args.use_distilbert, 'distilbert-base-german-cased' ))
                self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-german-cased")
            else :
                print('Dataset Name {}  Use DistilBert Flag  {} Model being used {}'.format(args.emoset, args.use_distilbert, 'dbmdz/bert-base-german-uncased' ))
                self.tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-german-uncased")
        else:
            print('using distill-uncased tokenizer')
            self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        
        
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
    
    '''
        Machine Learning models don’t work with text and require data preprocessing to convert text to numbers. As we are
        using BERT for modelling the word level inputs, we need to do following preprocessing

        - Add special tokens to separate sentences and do classification
        - Pass sequences of constant length (introduce padding)
        - Create array of 0s (pad token) and 1s (real token) called attention mask

        We will use the pretrained bert model and tokenizers from the Transformers Library 
    
    '''
    
    def transform_data(self, df, max_seq_len):
        """
        returns the padded input ids and attention masks according to the DistilBert tokenizer
        """
               
        def tokenize_fct(turn):
            return self.tokenizer.encode(turn, truncation = True, add_special_tokens=True, max_length=max_seq_len)
        
        if self.args.emoset in ['friends', 'emotionpush', 'emorynlp']:
            tokenized = df[['utterance']].applymap(tokenize_fct)
            padded = torch.tensor([[ids + [0]*(max_seq_len-len(ids)) for ids in idx] for idx in tokenized.values])
            attention_mask = torch.where(padded != 0, torch.ones_like(padded), torch.zeros_like(padded))
        elif self.args.emoset == 'semeval':
            tokenized = df[['turn1','turn2','turn3']].applymap(tokenize_fct)
            padded = torch.tensor([[ids + [0]*(max_seq_len-len(ids)) for ids in idx] for idx in tokenized.values])
            attention_mask = torch.where(padded != 0, torch.ones_like(padded), torch.zeros_like(padded))
        else :
            tokenized = df[['utterance_de_deepl']].applymap(tokenize_fct)
            padded = torch.tensor([[ids + [0]*(max_seq_len-len(ids)) for ids in idx] for idx in tokenized.values])
            attention_mask = torch.where(padded != 0, torch.ones_like(padded), torch.zeros_like(padded))
        return padded, attention_mask, tokenized

    
    def __getitem__(self, idx):
        
        if not self.args.emoset == 'semeval':
            assert idx  in list(range (len( self.data['dialogue_id'].unique())))
            df = self.data.loc[self.data.dialogue_id ==idx].reset_index(drop = True)
        else:
            df = self.data.loc[self.data.index ==idx].reset_index(drop = True)  
      
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
           speakers_array = np.array(df.speaker).reshape(-1, 1)
           spkr_emd = self.encoder.transform(speakers_array).toarray()
                
        else:
            spkr_emd = 0
        
        return padded, attention_mask, spkr_emd, labels
    




