import torch
import torch.nn as nn

# AutoGluon and HPO tools
import autogluon.core as ag
import pandas as pd
import numpy as np
import random
import math
from embedder import NLP_embedder
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.cluster import KMeans
import time
from data import load_data, SimpleDataset, load_wiki
from torch.utils.data import DataLoader
from plot import plot_TSNE_clustering
# Fixing seed for reproducibility
SEED = 999
random.seed(SEED)
np.random.seed(SEED)

ACTIVE_METRIC_NAME = 'accuracy'
REWARD_ATTR_NAME = 'objective'
datasets = [ "mnli","cola", "sst2", "mrpc","qqp", "rte"]#"qqp", "rte" 
eval_ds = [ "rtesmall", "qqpsmall","qqp", "rte"]


    
def calculate_word_freq(dataset, tokenizer):
    batch_size = 32
    vocab_size = 30500
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    interpretcount = torch.zeros(vocab_size).to(device)
    for i in range(math.ceil(len(dataset) / batch_size)):
                ul = min((i+1) * batch_size, len(dataset))
                batch_x = dataset[i*batch_size: ul]
                batch_x = tokenizer(batch_x, return_tensors="pt", padding=True, max_length = 256, truncation = True)
                x_ids = torch.flatten(batch_x["input_ids"].to(device))
                interpretcount.put_(x_ids, torch.ones(x_ids.shape, device = device) ,accumulate=True)
    return interpretcount


def train(args, config):

    torch.multiprocessing.set_start_method('spawn', force=True)
    max_epochs = int(config["DEFAULT"]["epochs"])

    batch_size = int(config["DEFAULT"]["batch_size"])

    dataset = config["DEFAULT"]["dataset"]
    print("dataset:", dataset)
    log_file = config["DEFAULT"]["directory"]+"/log_file.csv"

    
        
    print("running baseline")
    num_classes = 2
    if "mnli" in dataset:
        num_classes = 3
    num_classes2 = 2
    #  print("loading model")
    model = NLP_embedder(num_classes = num_classes,batch_size = batch_size)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    X_train, X_val, X_test, Y_train, Y_val, Y_test = load_data(name=dataset)

    model.generate_tree([X_train[2]])
    # print("analyzing", dataset)
    # torch.save(calculate_word_freq(X_train, model.tokenizer), config["DEFAULT"]["directory"]+"/baseword_freq.pt")


    model.analy = True
    embeded_wiki = model.embed(X_train)
   # print(embeded_wiki)
    torch.save(model.interpretcount, config["DEFAULT"]["directory"]+"/pretrained.pt")
   # print(model.interpretcount)
    model.reset()
#    print("loading dataset")
    


    print("training model on first dataset", dataset)
    model.analy = False
    model.fit(X_train, Y_train, X_val= X_val,Y_val= Y_val,  epochs=max_epochs)
    accuracy = float(model.evaluate(X_val,Y_val).cpu().numpy())

    model.analy = True
    embeded_task = model.embed(X_train)
    torch.save(model.interpretcount, config["DEFAULT"]["directory"]+"/afterfine.pt")
    model.reset()
    print("acuraccy on first ds:", accuracy)
    torch.cuda.empty_cache()
    #   print("loading dataset")
    # X_train, X_val2, _, Y_train, Y_val2, _ = load_data(name=dataset2)
    # print("training model  on second ds", dataset2)
    # model.fit(X_train, Y_train, epochs=max_epochs, second_head = True)
    # accuracy2 = float(model.evaluate(X_val2,Y_val2, second_head = True).cpu().numpy())
    # print("acuraccy on second ds:", accuracy2)
    
    # #   print("evaluating")
    # accuracy3 = float(model.evaluate(X_val,Y_val, second_head = False).cpu().numpy())
    # print("acuraccy on first ds after training on second ds:", accuracy3)
        

    torch.cuda.empty_cache()

