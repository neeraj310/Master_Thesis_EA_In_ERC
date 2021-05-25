#!/usr/bin/env python
# coding: utf-8

# # Problem Statement
# 
# 
# 
# 
# 
# ERC is a task that aims at predicting emotion of each utterance in a conversation. The following is an excerpt of a conversation with each utterance tagged with corresponding emotion and sentiment label.
# 
# ![alt text](example.jpg "Title")
# 

# # Hierarchical Transformer Network for Utterance-Level Emotion Recognition
# This is the Pytorch implementation Utterance-level Emotion Recognition [paper](https://arxiv.org/ftp/arxiv/papers/2002/2002.07551.pdf)

# # Overview
# 
# In this paper, we address four challenges in utterance-level emotion recognition in dialogue systems: 
# - Emotion depend on the context of previous utterances in the dialogue 
# - long-range contextual information is hard to be effectively captured; 
# - Datasets are quite small. 
# - We propose a hierarchical transformer framework with a lower-level transformer to model the word-level inputs and an upper-level transformer to capture the contexts of utterance-level embeddings. 
# 

# In[1]:


#Dataset


# In[2]:


import torch
import os
import random
import pandas as pd
import functools
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.distributed import DistributedSampler


# In[3]:



from ipywidgets import interact
import numpy as np
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc


# In[4]:


from transformers import DistilBertModel, DistilBertForSequenceClassification, DistilBertConfig, DistilBertTokenizer


# In[5]:


import time
import math
import argparse


# In[6]:


from collections import defaultdict
from sklearn.metrics import f1_score


# In[7]:


import pytorch_lightning as pl
from pytorch_lightning import seed_everything


# In[8]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# In[9]:


sns.set(style='whitegrid', palette='muted', font_scale=1.2)
HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]
sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))
rcParams['figure.figsize'] = 12, 8


# In[10]:


# Timer
def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


# In[11]:


path = 'data/clean_train.txt'
df = pd.read_csv(path, delimiter='\t', index_col='id')
df.head()


# In[12]:


df.shape


# In[13]:


df.info()


# In[14]:


df = df.iloc[0:2000, :]
df.shape


# ## Challenges 
#  - very unformal language, typos, and bad grammar. 
#  - unbalanced dataset:
#  

# In[15]:


df['label'].value_counts()


# In[16]:


sns.countplot(df.label)
plt.xlabel('review score');


# In[17]:


token_lens = []
for col in ['turn1', 'turn2', 'turn3']:
    for tokens in df[col]:
        token_lens.append(len(tokens))
        
sns.distplot(token_lens)
plt.xlim([0, 256]);
plt.xlabel('Token count');


# Most of the utterances seem to contain less than 80 tokens, but we’ll be on the safe side and choose a maximum length of 100.

# In[18]:


# export 
def transform_data(df, max_seq_len):
    """
    returns the padded input ids and attention masks according to the DistilBert tokenizer
    """
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    def tokenize_fct(turn):
        return tokenizer.encode(turn, truncation = True, add_special_tokens=True, max_length=max_seq_len)
    
    tokenized = df[['turn1','turn2','turn3']].applymap(tokenize_fct)
    padded = torch.tensor([[ids + [0]*(max_seq_len-len(ids)) for ids in idx] for idx in tokenized.values])
    attention_mask = torch.where(padded != 0, torch.ones_like(padded), torch.zeros_like(padded))
    return padded, attention_mask, tokenized


# In[19]:


def get_labels(df, emo_dict):
    """
    returns the labels according to the emotion dictionary
    """
    return torch.tensor([emo_dict[label] for label in df['label'].values])


# In[20]:


emo_dict = {'others': 0, 'sad': 1, 'angry': 2, 'happy': 3}
labels = torch.tensor([emo_dict[label] for label in df['label'].values])


# In[21]:


# export 
def dataloader(path, max_seq_len, batch_size, emo_dict, labels = True, eval = False):
    """
    Transforms the .csv data stored in `path` according to DistilBert features and returns it as a DataLoader 
    """
    df = pd.read_csv(path, delimiter='\t', index_col='id')
    #df = df.iloc[0:1000, :]
    print(df.shape)
    padded, attention_mask, tokenizer = transform_data(df, max_seq_len)
    
    if labels:
        dataset = TensorDataset(padded, attention_mask, get_labels(df, emo_dict))
    
    else:
        dataset = TensorDataset(padded, attention_mask)
      
    train_sampler = None

    return DataLoader(dataset, batch_size=batch_size, shuffle= not(eval) , 
                      sampler=train_sampler, num_workers=0),  df.shape[0]


# In[22]:


# export
class sentence_embeds_model(torch.nn.Module):
    """
    instantiates the pretrained DistilBert model and the linear layer
    """
    
    def __init__(self, dropout = 0.1):
        super(sentence_embeds_model, self).__init__()
        
        self.transformer = DistilBertModel.from_pretrained('distilbert-base-uncased', dropout=dropout, 
                                                           output_hidden_states=True)
        self.embedding_size = 2 * self.transformer.config.hidden_size
        
    def layerwise_lr(self, lr, decay):
        """
        returns grouped model parameters with layer-wise decaying learning rate
        """
        bert = self.transformer
        num_layers = bert.config.n_layers
        opt_parameters = [{'params': bert.embeddings.parameters(), 'lr': lr*decay**num_layers}]
        opt_parameters += [{'params': bert.transformer.layer[l].parameters(), 'lr': lr*decay**(num_layers-l+1)} 
                            for l in range(num_layers)]
        return opt_parameters
               
    def forward(self, input_ids = None, attention_mask = None, input_embeds = None):
        """
        returns the sentence embeddings
        """
        if input_ids is not None:
            input_ids = input_ids.flatten(end_dim = 1)
        if attention_mask is not None:
            attention_mask = attention_mask.flatten(end_dim = 1)
        output = self.transformer(input_ids = input_ids, 
                                  attention_mask = attention_mask, inputs_embeds = input_embeds)
    
        cls = output[0][:,0]
        hidden_mean = torch.mean(output[1][-1],1)
        sentence_embeds = torch.cat([cls, hidden_mean], dim = -1)
        
        return sentence_embeds.view(-1, 3, self.embedding_size)


# In[23]:


# export 
class context_classifier_model(torch.nn.Module):
    """
    instantiates the DisitlBertForSequenceClassification model, the position embeddings of the utterances, 
    and the binary loss function
    """
    
    def __init__(self, embedding_size, projection_size, n_layers, emo_dict, dropout = 0.1):
        super(context_classifier_model, self).__init__()
        
        self.projection_size = projection_size
        self.projection = torch.nn.Linear(embedding_size, projection_size)         
        self.position_embeds = torch.nn.Embedding(3, projection_size)
        self.norm = torch.nn.LayerNorm(projection_size)
        self.drop = torch.nn.Dropout(dropout)
    
        context_config = DistilBertConfig(dropout=dropout, 
                                dim=projection_size,
                                hidden_dim=4*projection_size,
                                n_layers=n_layers,
                                n_heads = 1,
                                num_labels=4)

        self.context_transformer = DistilBertForSequenceClassification(context_config)
        self.others_label = emo_dict['others']
        self.bin_loss_fct = torch.nn.BCEWithLogitsLoss()
        
    def bin_loss(self, logits, labels):
        """
        defined the additional binary loss for the `others` label
        """
        bin_labels = torch.where(labels == self.others_label, torch.ones_like(labels), 
                                 torch.zeros_like(labels)).float()
        bin_logits = logits[:, self.others_label]    
        return self.bin_loss_fct(bin_logits, bin_labels)

    def forward(self, sentence_embeds, labels = None):
        """
        returns the logits and the corresponding loss if `labels` are given
        """
        
        position_ids = torch.arange(3, dtype=torch.long, device=sentence_embeds.device)
        position_ids = position_ids.expand(sentence_embeds.shape[:2]) 
        position_embeds = self.position_embeds(position_ids)
        sentence_embeds = self.projection(sentence_embeds) + position_embeds 
        sentence_embeds = self.drop(self.norm(sentence_embeds))
        if labels is None:
            return self.context_transformer(inputs_embeds = sentence_embeds.flip(1), labels = labels)[0]
        
        else:
            output =  self.context_transformer(inputs_embeds = sentence_embeds.flip(1), labels = labels)
            loss = output[0]
            logits = output[1]
            return loss + self.bin_loss(logits, labels), logits


# In[24]:


# export
def metrics(loss, logits, labels):
    cm = torch.zeros((4,4), device = loss.device)
    preds = torch.argmax(logits, dim=1)
    acc = (labels == preds).float().mean()
    for label, pred in zip(labels.view(-1), preds.view(-1)):
        cm[label.long(), pred.long()] += 1
        
    tp = cm.diagonal()[1:].sum()
    fp = cm[:, 1:].sum() - tp
    fn = cm[1:, :].sum() - tp 
    return {'val_loss': loss, 'val_acc': acc, 'tp': tp, 'fp': fp, 'fn': fn}

def f1_score(tp, fp, fn):
    prec_rec_f1 = {}
    prec_rec_f1['precision'] = tp / (tp + fp)
    prec_rec_f1['recall'] = tp / (tp + fn)
    prec_rec_f1['f1_score'] = 2 * (prec_rec_f1['precision'] * prec_rec_f1['recall']) / (prec_rec_f1['precision'] + prec_rec_f1['recall'])
    return prec_rec_f1


# In[25]:


# export
class EmotionModel(pl.LightningModule):
    """
    PyTorch Lightning module for the Contextual Emotion Detection in Text Challenge
    """

    def __init__(self):
        """
        pass in parsed HyperOptArgumentParser to the model
        """
        super(EmotionModel, self).__init__()

        self.model = 'Semeval'
        self.emo_dict = {'others': 0, 'sad': 1, 'angry': 2, 'happy': 3}
        self.max_seq_len = args.max_seq_len
        self.dropout = 0.1
        self.projection_size = 100
        self.n_layers = 1
        self.sentence_embeds_model = sentence_embeds_model(dropout = self.dropout)
        self.context_classifier_model = context_classifier_model(self.sentence_embeds_model.embedding_size,
                                                                 self.projection_size, 
                                                                 self.n_layers, 
                                                                 self.emo_dict, 
                                                                 dropout = self.dropout)
        

    def forward(self, input_ids, attention_mask, labels = None):
        """
        no special modification required for lightning, define as you normally would
        """
        sentence_embeds = self.sentence_embeds_model(input_ids = input_ids, 
                                                             attention_mask = attention_mask)
        return self.context_classifier_model(sentence_embeds = sentence_embeds, labels = labels)
    
    def training_step(self, batch, batch_idx):
        """
        Lightning calls this inside the training loop
        """       
        input_ids, attention_mask, labels = batch
        #print(input_ids.shape)
        loss, _ = self.forward(input_ids = input_ids, attention_mask = attention_mask, labels = labels)
       
        tensorboard_logs = {'train_loss': loss}
        self.log('train_loss', loss, prog_bar=True)
        return {'loss': loss, 'log': tensorboard_logs}

    
    def validation_step(self, batch, batch_idx):
        """
        Lightning calls this inside the validation loop
        """
        input_ids, attention_mask, labels = batch
        loss, logits = self.forward(input_ids = input_ids, attention_mask = attention_mask, labels = labels)
        scores_dict = metrics(loss, logits, labels)

        return scores_dict
    
    def validation_epoch_end(self, outputs):
        """
        called at the end of validation to aggregate outputs
        :param outputs: list of individual outputs of each validation step
        :return:
        """
        
        #import IPython; IPython.embed();  exit(1)
        tqdm_dict = {}

        for metric_name in outputs[0].keys():
            metric_total = 0

            for output in outputs:
                metric_value = output[metric_name]
                metric_total += metric_value
            if metric_name in ['tp', 'fp', 'fn']:
                tqdm_dict[metric_name] = metric_total
            else:
                tqdm_dict[metric_name] = metric_total / len(outputs)

               
        prec_rec_f1 = f1_score(tqdm_dict['tp'], tqdm_dict['fp'], tqdm_dict['fn'])
        tqdm_dict.update(prec_rec_f1) 
        
        self.log('valid_loss', tqdm_dict["val_loss"], prog_bar=False)
        self.log('valid_acc', tqdm_dict["val_acc"], prog_bar=False)
        print('\nError Metric {}'.format(tqdm_dict))
        result = {'progress_bar': tqdm_dict, 'log': tqdm_dict, 'val_loss': tqdm_dict["val_loss"]}
      
        return result
        
         
    
    def test_step(self, batch, batch_idx):
        
        return self.validation_step(batch, batch_idx)
    
    def test_epoch_end(self, outputs):
        print('In test end \n\n\n\n')
        return self.validation_epoch_end(outputs)
    
    def configure_optimizers(self):
        """
        returns the optimizer and scheduler
        """
        params = model.sentence_embeds_model.layerwise_lr(args.lr, args.layerwise_decay)
        params += [{'params': model.context_classifier_model.parameters()}]
        self.optimizer = torch.optim.Adam(params, lr=args.lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=10)
        return [self.optimizer], [self.scheduler]
    

   
    def train_dataloader(self):
         train_loader, _ = dataloader(args.train_file, args.max_seq_len, args.batch_size, self.emo_dict, eval= False)
         return train_loader

    def val_dataloader(self):
        val_loader, _ =dataloader(args.val_file,  args.max_seq_len, args.batch_size, self.emo_dict, eval= True)
        return val_loader

    def test_dataloader(self):
        test_loader, _ =  dataloader(args.test_file, args.max_seq_len, args.batch_size, self.emo_dict, eval = True)
        return test_loader


# In[26]:


# export
def get_args():
    """
        returns the Parser args
    """
    root_dir = os.getcwd()  
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--projection_size', type=int, default=100)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--bert_lr', type=float, default=1e-5)
    parser.add_argument('--layerwise_decay', default=0.95, type=float,  
                    help='layerwise decay factor for the learning rate of the pretrained DistilBert')
    parser.add_argument('--train_file', default=os.path.join(root_dir, 'data/clean_train.txt'), type=str)
    parser.add_argument('--val_file', default=os.path.join(root_dir, 'data/clean_val.txt'), type=str)
    parser.add_argument('--test_file', default=os.path.join(root_dir, 'data/clean_test.txt'), type=str)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--epochs', type=int, default=3,
                        help='number of total epochs to run')
    parser.add_argument('--max_seq_len', type=int, default=100)
   
    parser.add_argument('--alpha', type=float, default=0.95)
    parser.add_argument('--seed', type=int, default=123,
                       help='seed for initializing training')
    parser.add_argument('--device', type=int, default=0)

    parser.add_argument('--evaluate', action='store_true')
    args = parser.parse_args(args=[])
    return args


# In[27]:


args = get_args()
seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

seed_everything(42, workers=True)
# In[28]:


model = EmotionModel()


# In[29]:



early_stop_callback = pl.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.00, patience=5, 
                                    verbose=True, mode='min')


trainer = pl.Trainer(default_root_dir=os.getcwd(),
                    gpus=(1 if torch.cuda.is_available() else 0),
                    max_epochs= 2,
                    fast_dev_run=False,
                    deterministic=True,
                    )
trainer.fit(model)



# In[30]:


trainer.test()


# In[ ]:




