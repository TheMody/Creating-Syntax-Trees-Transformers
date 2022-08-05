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
from torch_geometric.loader import DataLoader
from plot import plot_TSNE_clustering
from torch_geometric.data import Data
from sklearn.preprocessing import OneHotEncoder
# Fixing seed for reproducibility
SEED = 999
random.seed(SEED)
np.random.seed(SEED)

ACTIVE_METRIC_NAME = 'accuracy'
REWARD_ATTR_NAME = 'objective'
datasets = [ "mnli","cola", "sst2", "mrpc","qqp", "rte"]#"qqp", "rte" 
eval_ds = [ "rtesmall", "qqpsmall","qqp", "rte"]




def tree_to_graph(tree):
    def recursiv_tree_to_graph(tree,graph):
       # print(tree)
        wordpos,layer = tree.name 
       # print("tree",tree.name)
        posorg = wordpos *12 + layer
        for leaf in tree.get_children():
        #    print("leaf",leaf.name)
            wordpos,layer = leaf.name 
            pos = wordpos *12 + layer
           # if not pos in graph:
            graph[0].append(posorg)
            graph[1].append(pos)
            if not leaf.is_leaf():
              graph = recursiv_tree_to_graph(leaf,graph)
        return graph
    graph = ([],[])
    for child in tree.get_children():
        graph = recursiv_tree_to_graph(child,graph)
    return graph

def train(args, config):

    torch.multiprocessing.set_start_method('spawn', force=True)
    max_epochs = int(config["DEFAULT"]["epochs"])

    batch_size = int(config["DEFAULT"]["batch_size"])

    dataset = config["DEFAULT"]["dataset"]
    print("dataset:", dataset)
    log_file = config["DEFAULT"]["directory"]+"/log_file.csv"
    load = config["DEFAULT"]["load"] == "True"
    
        
    print("running baseline")
    num_classes = 2
    if "mnli" in dataset:
        num_classes = 3
    num_classes2 = 2
    #  print("loading model")

    def create_dataset(dataset):
        model = NLP_embedder(num_classes = num_classes,batch_size = batch_size)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        X_train, X_val, X_test, Y_train, Y_val, Y_test = load_data(name=dataset)

        model.fit(X_train, Y_train, X_val= X_val,Y_val= Y_val,  epochs=max_epochs)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        Y_train = Y_train.to(device)
        y_pred = model.predict(X_train)
        y  = (Y_train == y_pred)

        data_list = []
        for a,example in enumerate(X_train):
            edge_index = []
            x =[]
            tree = model.generate_tree([example])
            graph = tree_to_graph(tree)
            edge_index.append(torch.tensor([graph[0], graph[1]], dtype=torch.long))
            x_in = model.tokenizer([example], return_tensors="pt", padding=model.padding, max_length = 256, truncation = True)
            xvalues = torch.zeros((x_in["input_ids"].shape[1] * 12,256))
            for i in range(x_in["input_ids"].shape[1] * 12):
                xvalues[i,int(i / 12)] = 1
            x.append(xvalues)
            data = Data(x=x, y=y[a], edge_index=edge_index)
            data.num_nodes = x_in["input_ids"].shape[1] * 12
            data_list.append(data)
        torch.save(data_list, "sst2graphds.dt")

        return data_list

    if not load:
        data_list = create_dataset(dataset)
    else:
        data_list = torch.load("sst2graphds.dt")

  #  print(data_list)
    from Graph_network import GNNet

    def train_GNN(model, train_loader):
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
        crit = torch.nn.BCELoss()

        loss_all = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            output = model(data)
            label = data.y.type(torch.FloatTensor).to(device)
            loss = crit(output, label)
            loss.backward()
            loss_all += data.num_graphs * loss.item()
           # print("loss on batch",loss.item())
            optimizer.step()
        return loss_all / len(data_list)
        
    device = torch.device('cuda')
    model = GNNet().to(device)

    train_loader = DataLoader(data_list, batch_size=batch_size) 
    for epoch in range(3):
        loss = train_GNN(model, train_loader)
        print("average loss", loss)


        

    torch.cuda.empty_cache()

