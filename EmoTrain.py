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

def metrics(loss, logits, labels, class_weights, emo_dict,focus_emo, args):
    
    labelslist = labels.cpu().tolist()
    preds = torch.argmax(logits, dim=1)
    predslist = preds.cpu().tolist() 
  
    if 0:
        index = [i for i, j in enumerate(labelslist) if j in focus_emo ]
        tensor_index = torch.tensor(index, dtype=torch.int64)
        tensor_index = tensor_index.cuda(args.device)
        labels = labels.gather(0, tensor_index)
        preds = preds.gather(0, tensor_index)
   
      
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
    
    f1score = []
    f1_numenator = [2*x*y for (x, y) in zip(Recall, Precision)]
    f1_denominator = [x + y for (x, y) in zip(Recall, Precision)]

    for num1, num2 in zip(f1_numenator,f1_denominator):
        if  num2:
            f1score.append(np.round(num1 / num2, 2))
        else:
            f1score.append(0.0)
 
    prec_rec_f1['microF1_score'] = 2 * (prec_rec_f1['microPrecision'] * prec_rec_f1['microRecall']) / (prec_rec_f1['microPrecision'] + prec_rec_f1['microRecall'])
    prec_rec_f1['macroF1_score'] = np.round(sum(f1score) / len(f1score), 2)
    
    return prec_rec_f1

def metrics2(loss, logits, labels, emo_dict,focus_emo,args):
    TP = np.zeros([len(emo_dict.keys())], dtype=np.int64) # recall
    TP_FN = np.zeros([len(emo_dict.keys())], dtype=np.int64) # gold
    log_prob = F.log_softmax(logits, dim=1)
    emo_predidx = torch.argmax(log_prob, dim=1)
    emo_true = labels
   
    for lb in range(emo_true.size(0)):
        idx = emo_true[lb].item()
        TP_FN[idx] += 1
        if idx in focus_emo:
            if emo_true[lb] == emo_predidx[lb]:
                TP[idx] += 1

    f_TP = TP[focus_emo]
    f_TP_FN = TP_FN[focus_emo]

    return {'tp': f_TP, 'tp_fn': f_TP_FN}

def f1_score2(tp, tp_fn):
      
    prec_rec_f1 = {}
    Recall = [np.round(tp/tp_fn*100, 2) if tp_fn>0 else 0 for tp,tp_fn in zip(tp,tp_fn)]
    prec_rec_f1['Recall'] = Recall
    prec_rec_f1['wRecall'] = np.round((sum(tp)/ sum(tp_fn))*100, 2)
    prec_rec_f1['uRecall'] = sum(Recall) / len(Recall)

    return prec_rec_f1


# export
def metrics3(loss, logits, labels):
    cm = torch.zeros((4,4), device = loss.device)
    preds = torch.argmax(logits, dim=1)
    acc = (labels == preds).float().mean()
    for label, pred in zip(labels.view(-1), preds.view(-1)):
        cm[label.long(), pred.long()] += 1
        
    tp = cm.diagonal()[1:].sum()
    fp = cm[:, 1:].sum() - tp
    fn = cm[1:, :].sum() - tp 
    return {'val_loss': loss, 'val_acc': acc, 'tp': tp, 'fp': fp, 'fn': fn}

def f1_score3(tp, fp, fn):
    prec_rec_f1 = {}

    prec_rec_f1['precision'] = tp / (tp + fp)
    prec_rec_f1['recall'] = tp / (tp + fn)
    prec_rec_f1['f1_score'] = 2 * (prec_rec_f1['precision'] * prec_rec_f1['recall']) / (prec_rec_f1['precision'] + prec_rec_f1['recall'])
    return prec_rec_f1



def getMetrics(predictions, ground, emo_dict, focus_dict, args):
     

    label2emotion = {0:"others", 1:"happy", 2: "sad", 3:"angry"}
    #label2emotion = {0:"neutral", 1:"sadness", 2: "angry", 3:"happy"}
    
    discretePredictions = to_categorical(predictions, len(emo_dict.keys()))
    ground = to_categorical(ground, len(emo_dict.keys()))    
    truePositives = np.sum(discretePredictions*ground, axis=0)
    falsePositives = np.sum(np.clip(discretePredictions - ground, 0, 1), axis=0)
    falseNegatives = np.sum(np.clip(ground-discretePredictions, 0, 1), axis=0)
    
    print("True Positives per class : ", truePositives)
    print("False Positives per class : ", falsePositives)
    print("False Negatives per class : ", falseNegatives)
    
    # ------------- Macro level calculation ---------------
    macroPrecision = 0
    macroRecall = 0
    # We ignore the "Others" class during the calculation of Precision, Recall and F1
    if args.emoset == 'semeval':
        start = 1
        end = len(emo_dict.keys())
    elif args.emoset == 'semeval' or args.emoset == 'emotionpush':
        start = 0
        end = len(focus_dict)
    else :
        start = 0
        end = len(focus_dict)
    for c in range(start, end):
        precision = truePositives[c] / (truePositives[c] + falsePositives[c])
        macroPrecision += precision
        recall = truePositives[c] / (truePositives[c] + falseNegatives[c])
        macroRecall += recall
        f1 = ( 2 * recall * precision ) / (precision + recall) if (precision+recall) > 0 else 0
        print("Class %s : Precision : %.3f, Recall : %.3f, F1 : %.3f" % (label2emotion[c], precision, recall, f1))
    
    macroPrecision /= 3
    macroRecall /= 3
    macroF1 = (2 * macroRecall * macroPrecision ) / (macroPrecision + macroRecall) if (macroPrecision+macroRecall) > 0 else 0
    print("Ignoring the Others class, Macro Precision : %.4f, Macro Recall : %.4f, Macro F1 : %.4f" % (macroPrecision, macroRecall, macroF1))   
    
    # ------------- Micro level calculation ---------------
    truePositives = truePositives[start:end].sum()
    falsePositives = falsePositives[start:end].sum()
    falseNegatives = falseNegatives[start:end].sum()    
    
    print("Ignoring the Others class, Micro TP : %d, FP : %d, FN : %d" % (truePositives, falsePositives, falseNegatives))
    
    microPrecision = truePositives / (truePositives + falsePositives)
    microRecall = truePositives / (truePositives + falseNegatives)
    
    microF1 = ( 2 * microRecall * microPrecision ) / (microPrecision + microRecall) if (microPrecision+microRecall) > 0 else 0
    # -----------------------------------------------------
    
    #predictions = predictions.argmax(axis=1)
    ground = ground.argmax(axis=1)
    accuracy = np.mean(predictions==ground)
    
    print("Accuracy : %.4f, Micro Precision : %.4f, Micro Recall : %.4f, Micro F1 : %.4f" % (accuracy, microPrecision, microRecall, microF1))
    return accuracy, microPrecision, microRecall, microF1





class EmotionModel(pl.LightningModule):
    """
    PyTorch Lightning module for the Contextual Emotion Detection in Text Challenge
    """

    def __init__(self, emo_dict, focus_dict, args, df_train, df_val, df_test,  verbosity = 1):
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
            self.encoder, self.max_number_of_speakers_in_dialogue = self.get_speakers_encoder()
            if 0:
                self.max_number_of_speakers_in_dialogue = self.get_max_number_of_speakers_in_dialogue()
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

    
    def get_speakers_encoder(self):
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
    '''
    
    def get_speakers_encoder(self):
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
    '''     
    def calc_loss_weight(self, df, rate=1.0):
        """ Loss weights """
        emo_count = {}
        for emo in self.emo_dict.keys():
            emo_count[emo] =df['label'].value_counts()[emo]
        min_emo = float(min([ emo_count[w] for w in self.focus_dict]))
        weight = [math.pow(min_emo / emo_count[k], rate) if k in self.focus_dict else 0 for k,v in emo_count.items()]
        weight = np.array(weight)
        weight /= np.sum(weight)
        weight = torch.from_numpy(weight).float()
   
        return weight
    
    def calc_class_weight(self, df, rate=1.0):
        """ Loss weights """
        emo_count = {}
        for emo in self.emo_dict.keys():
            if emo in self.focus_dict:
                emo_count[emo] =df['label'].value_counts()[emo]
            
        
        weight = [(v / sum(emo_count.values())) for k,v in emo_count.items()]
        weight = np.array(weight)
        weight = torch.from_numpy(weight).float()
       
        return weight
        
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
        scores_dict2 = metrics2(loss, logits, labels,self.emo_dict, self.focus_emo, self.args)
        if self.args.emoset == 'semeval':
        	scores_dict3 = metrics3(loss, logits, labels)
        	return scores_dict1, scores_dict2, scores_dict3
        else:
            return scores_dict1, scores_dict2  
    def validation_epoch_end(self, outputs):
        """
        called at the end of validation to aggregate outputs
        :param outputs: list of individual outputs of each validation step
        :return:
        """
     
        tqdm_dict = {}
        tqdm_dict2 = {}
        tqdm_dict3 = {}
        
        tqdm_dict['preds'] = [0 for w in self.focus_dict]
        tqdm_dict['labels'] = [0 for w in self.focus_dict]
        tqdm_dict['tp'] = [0 for w in self.focus_dict]
        tqdm_dict['fp'] = [0 for w in self.focus_dict]
        tqdm_dict['fn'] = [0 for w in self.focus_dict]
        tqdm_dict['acc_per_class'] = [0 for w in self.focus_dict]
        tqdm_dict['y_pred'] = []
        tqdm_dict['y_true'] = []
        tqdm_dict['val_loss'] = 0
        
        tqdm_dict2['tp'] = [0 for w in self.focus_dict]
        tqdm_dict2['tp_fn'] = [0 for w in self.focus_dict ]
        
        for metric_name in outputs[0][0].keys():
          
            for output in outputs:
                metric_value = output[0][metric_name]
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
        
        for metric_name in outputs[0][1].keys():
            for output in outputs:
                metric_value = output[1][metric_name]
                tqdm_dict2[metric_name] += metric_value
               
        #import IPython; IPython.embed();  exit(1)
        
        '''
        if ( self.loop_count ==1):
            import IPython; IPython.embed();  exit(1)
        self.loop_count =  self.loop_count+1
        '''
        prec_rec_f1 = calc_f1_score(tqdm_dict['tp'], tqdm_dict['fp'], tqdm_dict['fn'])
        tqdm_dict.update(prec_rec_f1) 
        
        prec_rec_f1 = f1_score2(tqdm_dict2['tp'], tqdm_dict2['tp_fn'])
        tqdm_dict2.update(prec_rec_f1) 
        y_test = tqdm_dict.pop("y_true", None)
        y_pred = tqdm_dict.pop("y_pred", None)
       
    
    
       
        if self.args.emoset == 'semeval': 
            for metric_name in outputs[0][2].keys():
                metric_total = 0

                for output in outputs:
                    metric_value = output[2][metric_name]

                    if metric_name in ['tp', 'fp', 'fn']:
                        metric_value = torch.sum(metric_value)
                    else:
                        metric_value = torch.mean(metric_value)
                    
                metric_total += metric_value
                if metric_name in ['tp', 'fp', 'fn']:
                    tqdm_dict3[metric_name] = metric_total
                else:
                    tqdm_dict3[metric_name] = metric_total / len(outputs)

               
            prec_rec_f1 = f1_score3(tqdm_dict3['tp'], tqdm_dict3['fp'], tqdm_dict3['fn'])
            tqdm_dict3.update(prec_rec_f1) 
            getMetrics(np.asarray(y_pred), np.asarray(y_test), self.emo_dict, self.focus_dict, self.args ) 
            
                   
        self.log('valid_loss', tqdm_dict["val_loss"], prog_bar=False)
        self.log('valid_ac_unweighted', tqdm_dict["acc_unweighted"], prog_bar=False)
        self.log('macroRecall', tqdm_dict["macroRecall"], prog_bar=False)
        #print('\nError Metric {}'.format(tqdm_dict))
        if self.verbosity :
          
            print(*tqdm_dict.items(), sep='\n')
            print(*tqdm_dict2.items(), sep='\n')
            if self.args.emoset == 'semeval':
                print(*tqdm_dict3.items(), sep='\n')
            result = {'progress_bar': tqdm_dict, 'log': tqdm_dict, 'val_loss': tqdm_dict["val_loss"]}
        else:
            result = tqdm_dict['macroRecall']
        
       
        return result
        #return
    
    def test_step(self, batch, batch_idx):
 
        return self.validation_step(batch, batch_idx)

    
    def test_epoch_end(self, outputs):
        return_value =  self.validation_epoch_end(outputs)
        print('return value is {}'.format(return_value))
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
        print('f1 score is {}'.format(score[0]['macroRecall']))
    else:
        checkpoint_path = './lightning_logs/version_0/checkpoints/'
        checkpoints = os.listdir(checkpoint_path)

        for checkpoint in checkpoints:
            print(checkpoint)
            score = trainer.test(ckpt_path = checkpoint_path + checkpoint)
            print('f1 score is {}'.format(score[0]['macroRecall']))
    return