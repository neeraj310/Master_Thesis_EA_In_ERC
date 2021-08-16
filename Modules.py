# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 14:52:51 2021

@author: neera
"""
import math
import torch
import numpy as np
import torch.nn.functional as F
from transformers import DistilBertModel, DistilBertForSequenceClassification, DistilBertConfig
from transformers import AutoModel, AutoTokenizer
from transformers import AutoConfig, AutoModelForPreTraining, AutoModelForSequenceClassification

'''
  This class makes use of DistilBert model to model word level input and capture the context of utterance level embeddings.
  It makes use of pretrained language model which is equivalent to introducing external data into the model and helps our
  model obtain better utterance embedding
  Parameters
    ----------
    args: Structure of various flags used in the code. 
    droput : Dropout  
  
  Returns
    -------
  sentence_embeds : Tensor
     Tensor containing the utterance level embeddings.
     
'''


class sentence_embeds_model(torch.nn.Module):
    """
    instantiates the pretrained DistilBert model and the linear layer
    """
    
    def __init__(self, args, dropout = 0.1):
        super(sentence_embeds_model, self).__init__()
        self.args = args
        if args.emoset == 'friends_german':
            if args.use_distilbert:
                print('Dataset Name {}  Use DistilBert Flag  {} Model being used {}'.format(args.emoset, args.use_distilbert, 'distilbert-base-german-cased' ))
                self.transformer = DistilBertModel.from_pretrained('distilbert-base-german-cased', dropout=dropout, 
                                                           output_hidden_states=True)
            else:
                print('Dataset Name {}  Use DistilBert Flag  {} Model being used {}'.format(args.emoset, args.use_distilbert, 'dbmdz/bert-base-german-uncased' ))
                self.transformer = AutoModel.from_pretrained("dbmdz/bert-base-german-uncased", output_hidden_states=True)
                                                           
        else:
            self.transformer = DistilBertModel.from_pretrained('distilbert-base-uncased', dropout=dropout, 
                                                           output_hidden_states=True)
        self.embedding_size = 2 * self.transformer.config.hidden_size
     
      
        
    def layerwise_lr(self, lr, decay):
        """
        returns grouped model parameters with layer-wise decaying learning rate
        """
        bert = self.transformer
        
        if self.args.emoset == 'friends_german' and not self.args.use_distilbert:
            num_layers = bert.config.num_hidden_layers
            opt_parameters = [{'params': bert.embeddings.parameters(), 'lr': lr*decay**num_layers}]
            opt_parameters += [{'params': bert.encoder.layer[l].parameters(), 'lr': lr*decay**(num_layers-l+1)} 
                            for l in range(num_layers)]
        else:
            num_layers = bert.config.n_layers
            print('num_layers = {}'.format(num_layers))
            opt_parameters = [{'params': bert.embeddings.parameters(), 'lr': lr*decay**num_layers}]
            opt_parameters += [{'params': bert.transformer.layer[l].parameters(), 'lr': lr*decay**(num_layers-l+1)} 
                            for l in range(num_layers)]
     
        return opt_parameters
               
    def forward(self, input_ids = None, attention_mask = None, input_embeds = None):
        """
        returns the sentence embeddings
        """
        utterance_count = input_ids.shape[1]
        
        if input_ids is not None:
            input_ids = input_ids.flatten(end_dim = 1)
     
        if attention_mask is not None:
            attention_mask = attention_mask.flatten(end_dim = 1)
        
       
       
        output = self.transformer(input_ids = input_ids, 
                                  attention_mask = attention_mask, inputs_embeds = input_embeds)
      
        cls = output[0][:,0]
        if self.args.emoset == 'friends_german' and not self.args.use_distilbert:
            hidden_mean = torch.mean(output[2][-1],1)
        else:
            hidden_mean = torch.mean(output[1][-1],1)
        sentence_embeds = torch.cat([cls, hidden_mean], dim = -1)
        sentence_embeds = sentence_embeds.view(-1,utterance_count, self.embedding_size)
       
        return sentence_embeds
    
    
'''
  Create contextual sentence embeddings- We model that a conversation may consists of vatiable number of utterances for 
  dataset Friends/EmotionPush/EmoryNLP whereas for semeval number of utterances per conversation is fixed to 3. 
    ----------
    embedding_size: Embedded Layer Size
    projection_size :Size of fully connected lazer before the bert model. 
    n_layers : Number of hidden layers in pretrained bert model 
    emo_dict : Label Encoding
    max_number_of_speakers_in_dialogue
    max_number_of_utter_in_dialogue : 25 for Friends/EmotionPush/EmoryNLP and 3 for Semeval. 
    loss_weights : Tensor containing the weight assigned to different label classes to handle class imblance. 
  Returns
    -------
  logits : Tensor
     Softmax Probability Distribution

  loss : Scalor
      
'''
class context_classifier_model(torch.nn.Module):
    """
    instantiates the DisitlBertForSequenceClassification model, the position embeddings of the utterances, 
    and the binary loss function
    """
    
    def __init__(self, 
                 embedding_size, 
                 projection_size, 
                 n_layers, 
                 emo_dict, 
                 focus_dict, 
                 max_number_of_speakers_in_dialogue, 
                 max_number_of_utter_in_dialogue,
                 args,
                 loss_weights,
                 dropout = 0.1):
        super(context_classifier_model, self).__init__()
     
        self.projection_size = projection_size
        self.projection = torch.nn.Linear(embedding_size, projection_size)  
        self.max_number_of_speakers_in_dialogue = max_number_of_speakers_in_dialogue
        self.max_number_of_utter_in_dialogue = max_number_of_utter_in_dialogue
        self.position_embeds = torch.nn.Embedding( self.max_number_of_utter_in_dialogue, projection_size)
     
        self.norm = torch.nn.LayerNorm(projection_size+self.max_number_of_speakers_in_dialogue)
        self.drop = torch.nn.Dropout(dropout)
        self.focus_dict = focus_dict
        self.emo_dict = emo_dict
        self.args = args
        self.loss_weights = loss_weights
        context_config = DistilBertConfig(dropout=dropout, 
                                dim=projection_size+self.max_number_of_speakers_in_dialogue,
                                hidden_dim=4*projection_size,
                                n_layers=n_layers,
                                n_heads = 1,
                                num_labels=len(self.emo_dict.keys()))
      
        self.context_transformer = DistilBertForSequenceClassification(context_config)
    
        if not self.args.emoset == 'semeval':
            self.others_label = self.emo_dict['neutral']
        else:
            self.others_label = self.emo_dict['others']
        self.bin_loss_fct = torch.nn.BCEWithLogitsLoss()
        self.context_experiment_flag = False
   
        
    def bin_loss(self, logits, labels):
        """
        defined the additional binary loss for the `others` label
        """
        bin_labels = torch.where(labels == self.others_label, torch.ones_like(labels), 
                                 torch.zeros_like(labels)).float()
        bin_logits = logits[:, self.others_label]    
        return self.bin_loss_fct(bin_logits, bin_labels)

    
    
    def comput_loss(self, log_prob, target, weights):
        """ Weighted loss function """
        #loss = F.nll_loss(log_prob, target, weight=weights, reduction='sum')
        loss = F.cross_entropy(log_prob, target, weight=weights, reduction='sum')
        loss /= target.size(0)

        return loss


    def forward(self, sentence_embeds, spkr_emd, labels = None):
        """
        returns the logits and the corresponding loss if `labels` are given
        """
        if self.context_experiment_flag == True:
            sentence_embeds = sentence_embeds.transpose(0, 1)
            spkr_emd = spkr_emd.transpose(0,1)
            position_ids = torch.arange(sentence_embeds.shape[1], dtype=torch.long, device=sentence_embeds.device)
            
        else:
            position_ids = torch.arange(sentence_embeds.shape[1], dtype=torch.long, device=sentence_embeds.device)
    
        position_ids = position_ids.expand(sentence_embeds.shape[:2]) 
        position_embeds = self.position_embeds(position_ids)
        sentence_embeds = self.projection(sentence_embeds) + position_embeds 
        if self.args.speaker_embedding:
            sentence_embeds = torch.cat((sentence_embeds, spkr_emd), dim=2)
            sentence_embeds = sentence_embeds.to(device=self.args.device, dtype=torch.float)
        sentence_embeds = self.drop(self.norm(sentence_embeds))
        if labels is None:
            if not self.emoset == 'semeval':
                return self.context_transformer(inputs_embeds = sentence_embeds.transpose(0, 1), labels = labels)[0]
            else:
                return self.context_transformer(inputs_embeds = sentence_embeds.flip(1), labels = labels)[0]
        else:
            if not self.args.emoset == 'semeval':
                output =  self.context_transformer(inputs_embeds = sentence_embeds.transpose(0, 1), labels = labels)
            else:
                output =  self.context_transformer(inputs_embeds = sentence_embeds.flip(1), labels = labels)
        loss = output[0]
        logits = output[1]
        log_prob = logits
        if not self.args.emoset == 'semeval':
            loss2 = self.comput_loss(log_prob, labels, self.loss_weights)
            return loss2, logits
        else:
            return loss + self.bin_loss(logits, labels), logits
        