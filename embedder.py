from sqlalchemy import true
from transformers import BertTokenizer, BertModel, ElectraTokenizer, ElectraModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import time
import numpy as np
from transformers.utils import logging
from transformers import glue_convert_examples_to_features, DataCollatorForLanguageModeling
from copy import deepcopy
from torch.autograd import variable
from torch.utils.data import DataLoader
from data import load_wiki
import os
from transformers import BertTokenizer, BertForMaskedLM
logging.set_verbosity_error()

models = ['bert-base-uncased',
         'google/electra-small-discriminator',
         'distilbert-base-uncased',
         'gpt2'
          ]

class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor
    
class CosineScheduler(optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, max_iters):
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        return lr_factor


    


class NLP_embedder(nn.Module):

    def reset(self):
     #   self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    #    self.model = BertForMaskedLM.from_pretrained('bert-base-uncased')
        self.interpretcount = torch.zeros(self.interpretcount.shape)
        self.register_buffer("count", self.interpretcount)

        return

    def __init__(self,  num_classes, batch_size, lr= 2e-5):
        super(NLP_embedder, self).__init__()
        self.type = 'nn'
        self.batch_size = batch_size
        self.padding = True
        self.bag = False
        self.num_classes = num_classes
        self.lasthiddenstate = 0
        
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForMaskedLM.from_pretrained('bert-base-uncased')
        self.output_length = 768
        self.importance = 675
        vocab_size = 30500
        num_layers = 12
        num_heads = 12
        self.interpretcount = torch.zeros((num_layers, num_heads,vocab_size))
        self.register_buffer("count", self.interpretcount)
        self.analy = True
        
#         from transformers import RobertaTokenizer, RobertaModel
#         self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
#         self.model = RobertaModel.from_pretrained('roberta-base')
#         self.output_length = 768

        
        self.fc1 = nn.Linear(self.output_length,self.num_classes)
        self.criterion = torch.nn.CrossEntropyLoss()
        

        self.optimizer =optim.Adam(self.parameters(), lr=lr)
        self.softmax = torch.nn.Softmax(dim=1)
    
    def fitmlm(self,dataset, epochs):
        dataset = load_wiki()
        self.scheduler =CosineWarmupScheduler(optimizer= self.optimizer[i], 
                                               warmup = math.ceil(len(dataset)*epochs *0.1 / self.batch_size) ,
                                                max_iters = math.ceil(len(dataset)*epochs  / self.batch_size))


        for i in range(math.ceil(len(dataset) / self.batch_size)):
              #  batch_x, batch_y = next(iter(data))
           # start = time.time()
            ul = min((i+1) * self.batch_size, len(self.dataset))
            batch_x = self.dataset[i*self.batch_size: ul]
            self.model.zero_grad()
            labels = self.tokenizer(batch_x, return_tensors="pt", padding=True, max_length = 256, truncation = True)["input_ids"]
            input = self.tokenizer(batch_x, return_tensors="pt", padding=True, max_length = 256, truncation = True)
            input["input_ids"] = self.mask(input["input_ids"])
            labels = torch.where(input["input_ids"] == self.tokenizer.mask_token_id, labels, -100)
           # print("processing inputs took:",time.time()-start)
           # start = time.time()
            output = self.model(**input, labels = labels)
           # print("forward pass took:",time.time()-start)
            #start = time.time()
            loss = output.loss
            loss.backward()

            self.optimizer.step()

#                 if i % np.max((1,int((len(x)/self.batch_size)*0.001))) == 0:
#                     print(i, loss.item())
               # print(y_pred, batch_y)
            self.scheduler.step()
          #  print("backward pass took:",time.time()-start)
          #  start = time.time()
           # print("matrix add pass took:",time.time()-start)
            if i % 10 == 0:
                print("at", ul , "of", len(self.dataset))
        
    def forward(self, x_in):
        x = self.model(**x_in,output_attentions= True,output_hidden_states = True)   
        if self.analy:
            self.analyze(x.attentions, x_in)
        x = x.hidden_states[-1]
        x = x[:, self.lasthiddenstate]
        x = self.fc1(x)
        x = self.softmax(x)
        return x
    
    def analyze(self,attentions, x_in):
     #   start = time.time()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.interpretcount = self.interpretcount.to(device)
        for i,attention in enumerate(attentions):
            #   print(x_in["input_ids"].shape)
        #     print(attention > 0.5)
                attention_collapsed = torch.sum(attention > 0.5, dim = 3) >= 1
               # print(attention_collapsed.shape)
            #   print(attention_collapsed)
                attention_collapsed = torch.permute(attention_collapsed, (1,0,2))
              #  print(attention_collapsed.shape)
                for a,head in enumerate(attention_collapsed):
                    selected_input_ids = torch.masked_select(x_in["input_ids"], head)
                    self.interpretcount[i,a].put_(selected_input_ids, torch.ones(selected_input_ids.shape, device = device) ,accumulate=True)
     #   print(time.time()-start)
    
     
    def fit(self, x, y, epochs=1, X_val= None,Y_val= None, reporter = None):
        
        self.scheduler =CosineWarmupScheduler(optimizer= self.optimizer, 
                                               warmup = math.ceil(len(x)*epochs *0.1 / self.batch_size) ,
                                                max_iters = math.ceil(len(x)*epochs  / self.batch_size))

        


        accuracy = None
        for e in range(epochs):
            start = time.time()
            for i in range(math.ceil(len(x) / self.batch_size)):
              #  batch_x, batch_y = next(iter(data))
                ul = min((i+1) * self.batch_size, len(x))
                batch_x = x[i*self.batch_size: ul]
                batch_y = y[i*self.batch_size: ul]
           #     batch_x = glue_convert_examples_to_features(, tokenizer, max_length=128,  task=task_name)
                batch_x = self.tokenizer(batch_x, return_tensors="pt", padding=self.padding, max_length = 256, truncation = True)
             #   print(batch_x["input_ids"].size())
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                batch_y = batch_y.to(device)
                batch_x = batch_x.to(device)
                self.optimizer.zero_grad()
                y_pred = self(batch_x)
                loss = self.criterion(y_pred, batch_y)    
                loss.backward()
                self.optimizer.step()

#                 if i % np.max((1,int((len(x)/self.batch_size)*0.001))) == 0:
#                     print(i, loss.item())
               # print(y_pred, batch_y)
                self.scheduler.step()
            if X_val != None:
                with torch.no_grad():
                    accuracy = self.evaluate(X_val, Y_val)
                    print("accuracy after", e, "epochs:", float(accuracy.cpu().numpy()), "time per epoch", time.time()-start)
                    if reporter != None:
                        reporter(objective=float(accuracy.cpu().numpy()) / 2.0, epoch=e+1)
            else:
                print("epoch",e,"time per epoch", time.time()-start)
                
                

        return
    
    def evaluate(self, X,Y):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        Y = Y.to(device)
        y_pred = self.predict(X)
        accuracy = torch.sum(Y == y_pred)
        accuracy = accuracy/Y.shape[0]
        return accuracy
    
    def predict(self, x):
        resultx = None

        for i in range(math.ceil(len(x) / self.batch_size)):
            ul = min((i+1) * self.batch_size, len(x))
            batch_x = x[i*self.batch_size: ul]
            batch_x = self.tokenizer(batch_x, return_tensors="pt", padding=self.padding, max_length = 256, truncation = True)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            batch_x = batch_x.to(device)
            batch_x = self(batch_x)
            if resultx is None:
                resultx = batch_x.detach()
            else:
                resultx = torch.cat((resultx, batch_x.detach()))

     #   resultx = resultx.detach()
        return torch.argmax(resultx, dim = 1)
    
    def embed(self, x):
        resultx = None
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        for i in range(math.ceil(len(x) / self.batch_size)):
            ul = min((i+1) * self.batch_size, len(x))
            batch_x = x[i*self.batch_size: ul]
            batch_x = self.tokenizer(batch_x, return_tensors="pt", padding=self.padding, max_length = 256, truncation = True)
            batch_x_in = batch_x.to(device)
            batch_x  = self.model(**batch_x_in,output_attentions= True,output_hidden_states = True)  
            self.analyze(batch_x.attentions, batch_x_in)
            batch_x = batch_x.hidden_states[-1]
            batch_x = batch_x[:, self.lasthiddenstate]
            if resultx is None:
                resultx = batch_x.detach()
            else:
                resultx = torch.cat((resultx, batch_x.detach()))

     #   resultx = resultx.detach()
        return resultx
    
    

        
        

