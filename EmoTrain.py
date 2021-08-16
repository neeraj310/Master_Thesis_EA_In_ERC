# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 15:11:39 2021

@author: neera
"""
import Modules
import torch
import math
import numpy as np
import pytorch_lightning as pl
import CustomDataloader
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import keras
from keras.utils import to_categorical
import os
from sklearn.preprocessing import OneHotEncoder
from collections import Counter
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def metrics(loss, logits, labels, class_weights, emo_dict,focus_emo, args):
    
    labelslist = labels.cpu().tolist()
    preds = torch.argmax(logits, dim=1)
    predslist = preds.cpu().tolist() 
  

      
    cm = np.zeros((len(emo_dict.keys()),len(emo_dict.keys())), dtype=np.int64) # recall
       
  
    for label, pred in zip(labels.view(-1), preds.view(-1)):
        cm[label.long(), pred.long()] += 1
   
    if not args.emoset == 'semeval':
        cm = cm[0:len(focus_emo), 0:len(focus_emo)]
        gt_labels_per_class =  cm.sum(axis = 1)
        preds_per_class =  cm.sum(axis = 0)
        tp = cm.diagonal()[0:]
        fp = preds_per_class-tp
        fn = gt_labels_per_class- tp
    else:
        gt_labels_per_class =  cm[1:, :].sum(axis = 1)
        preds_per_class =  cm[:, 1:].sum(axis = 0)
        tp = cm.diagonal()[1:]
        fp = preds_per_class-tp
        fn = gt_labels_per_class- tp
             
      
    return {'val_loss': loss, 
            'tp': tp, 
            'fp': fp, 
            'fn': fn, 
            'preds':preds_per_class,
            'labels': gt_labels_per_class,
            'y_pred':predslist,
            'y_true':labelslist
            }

def calc_f1_score(tp, fp, fn):
    prec_rec_f1 = {}
    tp_fn = tp+fn
    tp_fp = tp+fp


    Recall = [np.round(tp/tp_fn*100, 2) if tp_fn>0 else 0.0 for tp,tp_fn in zip(tp,tp_fn)]
    prec_rec_f1['microRecall'] = np.round((sum(tp)/ sum(tp_fn))*100, 2)
    prec_rec_f1['macroRecall'] = np.round (sum(Recall) / len(Recall),2)
    
    Precision = [np.round(tp/tp_fp*100, 2) if tp_fp>0 else 0.0 for tp,tp_fp in zip(tp,tp_fp)]
    prec_rec_f1['microPrecision'] = np.round((sum(tp)/ sum(tp_fp))*100, 2)
    prec_rec_f1['macroPrecision'] = np.round(sum(Precision) / len(Precision),2)
    
    per_class_f1score = []
    f1_numenator = [2*x*y for (x, y) in zip(Recall, Precision)]
    f1_denominator = [x + y for (x, y) in zip(Recall, Precision)]

    for num1, num2 in zip(f1_numenator,f1_denominator):
        if  num2:
            per_class_f1score.append(np.round(num1 / num2, 2))
        else:
            per_class_f1score.append(0.0)
             
    #import IPython; IPython.embed();  exit(1)
    macroPrecision = prec_rec_f1['macroPrecision']
    macroRecall = prec_rec_f1['macroRecall']
    macroF1 = (2 * macroRecall * macroPrecision ) / (macroPrecision + macroRecall) if (macroPrecision+macroRecall) > 0 else 0
    
    prec_rec_f1['microF1_score'] = 2 * (prec_rec_f1['microPrecision'] * prec_rec_f1['microRecall']) / (prec_rec_f1['microPrecision'] + prec_rec_f1['microRecall'])
    prec_rec_f1['macroF1_score'] = np.round(macroF1, 2)
    prec_rec_f1['macroF1_per_class'] = per_class_f1score
    
    return prec_rec_f1




class EmotionModel(pl.LightningModule):
    """
    PyTorch Lightning module for the Contextual Emotion Detection in Text Challenge
    """

    def __init__(self, emo_dict, focus_dict, args, df_train, df_val, df_test, verbosity = 1):
        """
        pass in parsed HyperOptArgumentParser to the model
        """
        super(EmotionModel, self).__init__()

        self.model_name = 'Semeval'
        self.emo_dict = emo_dict
        self.focus_dict = focus_dict
        self.focus_emo = [emo_dict[w] for w in focus_dict]
        self.max_seq_len = args.max_seq_len
        self.dropout = args.dropout
        self.projection_size = args.projection_size
        self.args = args
        self.n_layers = args.n_layers
        self.df_train = df_train
        self.df_val = df_val
        self.df_test = df_test
        self.verbosity = verbosity
        if args.speaker_embedding:
            print('Speaker Embedding is Enabled')
            self.encoder, self.max_number_of_speakers_in_dialogue = self.get_speakers_encoder_with_complete_speakers_list()
            print('Max number of speakers in a dialgue = {}'.format(self.max_number_of_speakers_in_dialogue))
        else:
            print('Speaker Embedding is Disabled')
            self.max_number_of_speakers_in_dialogue =0
            self.encoder = 0
            
        if not self.args.emoset == 'semeval':
            self.max_number_of_utter_in_dialogue = self.get_max_utterance_length_in_dialogue()
        else:
            self.max_number_of_utter_in_dialogue = 3
        
        #Need to explore this hyperparameter
        self.weight_rate =1
        self.loss_weights = self.calc_loss_weight(self.df_train, self.weight_rate )
        self.loss_weights = self.loss_weights.cuda(self.args.device)
        print( 'loss weights = {}'.format(self.loss_weights))
        self.class_weights = self.calc_class_weight(self.df_train, self.weight_rate )
        self.class_weights = self.class_weights.cuda(self.args.device)
        print( 'class weights = {}'.format(self.class_weights))
        self.sentence_embeds_model = Modules.sentence_embeds_model(self.args, dropout = self.dropout)
        self.context_classifier_model = Modules.context_classifier_model(self.sentence_embeds_model.embedding_size,
                                                                 self.projection_size, 
                                                                 self.n_layers, 
                                                                 self.emo_dict, 
                                                                 self.focus_dict,
                                                                 self.max_number_of_speakers_in_dialogue,
                                                                 self.max_number_of_utter_in_dialogue,
                                                                 self.args,
                                                                 self.loss_weights,
                                                                 dropout = self.dropout)
        self.sentence_embeds_model.cuda(self.args.device)
        self.context_classifier_model.cuda(self.args.device)
    
    def calc_loss_weight(self, df, rate= 0.75):
        """ Loss weights """
        emo_count = {}
        for emo in self.emo_dict.keys():
            emo_count[emo] =df['label'].value_counts()[emo]
            
        min_emo = float(min([ emo_count[w] for w in self.focus_dict]))
        # Check whether balanced weight_warming is enabled and initialize weights accordingly
        if self.args.balance_weight_warming:
            weight = [math.pow(min_emo / emo_count[k], rate) if k in self.focus_dict else 0 for k,v in emo_count.items()]
        else:
            weight = [math.pow(1, rate) if k in self.focus_dict else 0 for k,v in emo_count.items()]
        weight = np.array(weight)
        weight /= np.sum(weight)
        weight = torch.from_numpy(weight).float()
        return weight
    
    def calc_class_weight(self, df, rate=1.0):
        """ class weights """
        emo_count = {}
        for emo in self.emo_dict.keys():
            if emo in self.focus_dict:
                emo_count[emo] =df['label'].value_counts()[emo]
            
        
        weight = [(v / sum(emo_count.values())) for k,v in emo_count.items()]
        weight = np.array(weight)
        weight = torch.from_numpy(weight).float()
        return weight
    
    
    def get_speakers_encoder_with_complete_speakers_list(self):
        speaker_list = list(self.df_train.speaker)
        speaker_list.extend(list(self.df_val.speaker))
        speaker_list.extend(list(self.df_test.speaker))
        speaker_set = set(speaker_list )
        speaker_list = []
        for e in speaker_set:
            templist = []
            templist.append(e)
            speaker_list.append(templist)
        cat = OneHotEncoder()
        cat.fit(np.asarray(speaker_list, dtype = object))
        return cat, len(speaker_set)
 
    def get_speakers_encoder_with_only_main_speakers_list(self):
        
        speaker_list = []
        complete_speaker_list = list(self.df_train['speaker'])
        complete_speaker_list.extend(list(self.df_val['speaker']))
        complete_speaker_list.extend(list(self.df_test['speaker']))
        occurence_count = Counter(complete_speaker_list)
        common_speakers = occurence_count.most_common(6)
        for e in common_speakers:
            templist = []
            templist.append(e[0])
            speaker_list.append(templist)
        speaker_list.append(['Unknown'])
        cat = OneHotEncoder()
        cat.fit(np.asarray(speaker_list, dtype = object))
        return cat, len(cat.categories_[0].tolist())

    def get_max_number_of_speakers_in_dialogue(self):
        number_of_speakers_in_dialogue = []
        train_dialogue_id_list = list((self.df_train['dialogue_id'].unique()))
        val_dialogue_id_list = list((self.df_val['dialogue_id'].unique()))
        test_dialogue_id_list = list((self.df_test['dialogue_id'].unique()))
        
        for dialogue in train_dialogue_id_list:
            number_of_speakers_in_dialogue.append(len(set(self.df_train[(self.df_train.dialogue_id ==dialogue)].speaker)))
        for dialogue in val_dialogue_id_list:
            number_of_speakers_in_dialogue.append(len(set(self.df_val[(self.df_val.dialogue_id ==dialogue)].speaker)))
        for dialogue in test_dialogue_id_list:
            number_of_speakers_in_dialogue.append(len(set(self.df_test[(self.df_test.dialogue_id ==dialogue)].speaker)))
        return( max(number_of_speakers_in_dialogue))
    
    def get_max_utterance_length_in_dialogue(self):
  
        train_dialogue_id_list = list((self.df_train['dialogue_id'].unique()))
        val_dialogue_id_list = list((self.df_val['dialogue_id'].unique()))
        test_dialogue_id_list = list((self.df_test['dialogue_id'].unique()))
        number_of_utterance_in_dialogue = []
        for dialogue in train_dialogue_id_list:
             number_of_utterance_in_dialogue.append((self.df_train.dialogue_id ==dialogue).sum())
        for dialogue in val_dialogue_id_list:
             number_of_utterance_in_dialogue.append((self.df_val.dialogue_id ==dialogue).sum())
        for dialogue in test_dialogue_id_list:
             number_of_utterance_in_dialogue.append((self.df_test.dialogue_id ==dialogue).sum())
        return( max(number_of_utterance_in_dialogue))
       

    def forward(self, input_ids, attention_mask, spkr_emd, labels = None):
        """
        no special modification required for lightning, define as you normally would
        """
      
        sentence_embeds = self.sentence_embeds_model(input_ids = input_ids, 
                                                             attention_mask = attention_mask)
    
        return self.context_classifier_model(sentence_embeds = sentence_embeds,spkr_emd = spkr_emd, labels = labels)
    
    def training_step(self, batch, batch_idx):
        """
        Lightning calls this inside the training loop
        """       
        input_ids, attention_mask, spkr_emd , labels = batch
        spkr_emd = spkr_emd.cuda(self.args.device)
        labels = labels.view(-1)
              
        loss, _ = self.forward(input_ids = input_ids, attention_mask = attention_mask, spkr_emd = spkr_emd, labels = labels)
       
        tensorboard_logs = {'train_loss': loss}
        self.log('train_loss', loss, prog_bar=True)
        return {'loss': loss, 'log': tensorboard_logs}

    
    def validation_step(self, batch, batch_idx):
        """
        Lightning calls this inside the validation loop
        """
        
        input_ids, attention_mask, spkr_emd , labels = batch
        spkr_emd = spkr_emd.cuda(self.args.device)
        labels = labels.view(-1)
        
        loss, logits = self.forward(input_ids = input_ids, attention_mask = attention_mask,spkr_emd = spkr_emd,  labels = labels)
        scores_dict1 = metrics(loss, logits, labels, self.class_weights, self.emo_dict,self.focus_emo, self.args)
      
        return scores_dict1
    def validation_epoch_end(self, outputs):
        """
        called at the end of validation to aggregate outputs
        :param outputs: list of individual outputs of each validation step
        :return:
        """
     
        tqdm_dict = {}
       
        tqdm_dict['preds'] = [0 for w in self.focus_dict]
        tqdm_dict['labels'] = [0 for w in self.focus_dict]
        tqdm_dict['tp'] = [0 for w in self.focus_dict]
        tqdm_dict['fp'] = [0 for w in self.focus_dict]
        tqdm_dict['fn'] = [0 for w in self.focus_dict]
        tqdm_dict['acc_per_class'] = [0 for w in self.focus_dict]
        tqdm_dict['y_pred'] = []
        tqdm_dict['y_true'] = []
        tqdm_dict['val_loss'] = 0
        
               
        for metric_name in outputs[0].keys():
            
          
            for output in outputs:
                metric_value = output[metric_name]
                if metric_name in ['y_pred', 'y_true']:
                    tqdm_dict[metric_name].extend(metric_value)
                else:
                    tqdm_dict[metric_name] += metric_value
                    
            if metric_name in ['val_loss']:
                tqdm_dict[metric_name] =  tqdm_dict[metric_name] / len(outputs)
                
       
             
        for i in range(len(self.focus_dict)):
            if  tqdm_dict['labels'][i]:
                tqdm_dict['acc_per_class'][i] = np.round((tqdm_dict['tp'][i] / tqdm_dict['labels'][i])*100, 2)
            else:
                tqdm_dict['acc_per_class'][i] = 0.0
        
        tqdm_dict['acc_unweighted'] = np.round(sum(tqdm_dict['acc_per_class']) / len(tqdm_dict['acc_per_class']),2)       
        tqdm_dict['acc_weighted'] = sum(weight * value for weight, value in zip(self.class_weights, tqdm_dict['acc_per_class']))
        
        tqdm_dict['acc_weighted'] = (tqdm_dict['acc_weighted'] * 10**2).round() / (10**2)
        
       
               

       
        prec_rec_f1 = calc_f1_score(tqdm_dict['tp'], tqdm_dict['fp'], tqdm_dict['fn'])
        tqdm_dict.update(prec_rec_f1) 
        
        self.y_true = tqdm_dict.pop("y_true", None)
        self.y_pred = tqdm_dict.pop("y_pred", None)
                 
      
        self.log('valid_loss', tqdm_dict["val_loss"], prog_bar=False)
        self.log('valid_ac_unweighted', tqdm_dict["acc_unweighted"], prog_bar=False)
        self.log('macroF1', tqdm_dict["macroF1_score"], prog_bar=False)
        self.log('microF1', tqdm_dict["microF1_score"], prog_bar=False)
       
        microF1_per_class = {}
        for emotion_class in range(len(tqdm_dict["macroF1_per_class"])):
            microF1_per_class[str(emotion_class)] = tqdm_dict["macroF1_per_class"][emotion_class]
        print('Hello')
        print(microF1_per_class)
        print('\n\n\n')
        d =  {'acc': 33, 'recall': 55}
        self.log('performance',microF1_per_class)
           
        if self.verbosity :
          
            print(*tqdm_dict.items(), sep='\n')
         
        result = {'progress_bar': tqdm_dict, 'log': tqdm_dict, 'val_loss': tqdm_dict["val_loss"]}
        return 
    
    def test_step(self, batch, batch_idx):
 
        return self.validation_step(batch, batch_idx)

    
    def test_epoch_end(self, outputs):
        return_value =  self.validation_epoch_end(outputs)
        return (return_value)
    
    def configure_optimizers(self):
        """
        returns the optimizer and scheduler
        """
        params = self.sentence_embeds_model.layerwise_lr(self.args.lr, self.args.layerwise_decay)
        params += [{'params': self.context_classifier_model.parameters()}]
        self.optimizer = torch.optim.Adam(params, lr=self.args.lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=10)
        return [self.optimizer], [self.scheduler]
    

   
    def train_dataloader(self):
        
        dataset = CustomDataloader.CustomDataset(self.df_train, self.max_number_of_speakers_in_dialogue, self.emo_dict, self.encoder, self.args)
        train_loader = DataLoader(dataset, self.args.batch_size, shuffle=True, num_workers=0)
      
        return train_loader

    

    def val_dataloader(self):

        dataset = CustomDataloader.CustomDataset(self.df_val, self.max_number_of_speakers_in_dialogue, self.emo_dict, self.encoder, self.args)
        val_loader = DataLoader(dataset, self.args.batch_size, shuffle=False, num_workers=0)
      
        return val_loader

    def test_dataloader(self):
        
        dataset = CustomDataloader.CustomDataset(self.df_test, self.max_number_of_speakers_in_dialogue, self.emo_dict, self.encoder, self.args)
        test_loader = DataLoader(dataset, self.args.batch_size, shuffle=False, num_workers=0)
       
        return test_loader     

def train_model (model):
    
    early_stop_callback = pl.callbacks.EarlyStopping(monitor='valid_loss', min_delta=0.0005, patience=5 ,
                                        verbose=True, mode='min')
    
    
    if model.args.emoset == 'emotionpush' or model.args.emoset == 'emotionpush_german' :
        checkpoint_callback = pl.callbacks.ModelCheckpoint( monitor='macroF1' , mode = 'max', save_top_k = model.args.epochs) 
    else:
        checkpoint_callback = pl.callbacks.ModelCheckpoint( monitor='valid_loss' , mode = 'min', save_top_k = model.args.epochs)

  
    gpu_list = [int(model.args.gpu)]  
    trainer = pl.Trainer(default_root_dir=os.getcwd(),
                    gpus=(gpu_list if torch.cuda.is_available() else 0),
                    max_epochs= model.args.epochs,
                    fast_dev_run=False,
                    deterministic=True,
                    callbacks = [early_stop_callback,checkpoint_callback] ,
                    )
    trainer.fit(model)
    return trainer
    
def test_model(trainer, model):
    if 1:
        score = trainer.test(verbose = model.verbosity)
       
    else:
        checkpoint_path = './lightning_logs/version_0/checkpoints/'
        checkpoints = os.listdir(checkpoint_path)

        for checkpoint in checkpoints:
            print(checkpoint)
            score = trainer.test(ckpt_path = checkpoint_path + checkpoint)
            
    return score
