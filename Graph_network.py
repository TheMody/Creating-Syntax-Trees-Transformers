embed_dim = 256
import torch
import torch.nn as nn
from torch_geometric.nn import TopKPooling, GINConv
from torch_geometric.nn import global_mean_pool,global_max_pool
import torch.nn.functional as F
class GNNet(torch.nn.Module):
    def __init__(self):
        super(GNNet, self).__init__()

        self.conv1 = GINConv(nn.Sequential(
          nn.Linear(embed_dim, 256),
          nn.ReLU(),
        ))
        self.conv2 = GINConv(nn.Sequential(
          nn.Linear(256, 256),
          nn.ReLU(),
        ))
        self.conv3 = GINConv(nn.Sequential(
          nn.Linear(256, 256),
          nn.ReLU(),
        ))
        self.conv4 = GINConv(nn.Sequential(
          nn.Linear(256, 256),
          nn.ReLU(),
        ))
        self.conv5 = GINConv(nn.Sequential(
          nn.Linear(256, 256),
          nn.ReLU(),
        ))
        self.lin1 = torch.nn.Linear(512, 256)
        self.lin2 = torch.nn.Linear(256, 64)
        self.lin3 = torch.nn.Linear(64, 1)
        self.bn1 = torch.nn.BatchNorm1d(256)
        self.bn2 = torch.nn.BatchNorm1d(64)
        self.act1 = torch.nn.ReLU()
        self.act2 = torch.nn.ReLU()        
  
    def forward(self, data):
        x, edge_index , batch = data.x[0], data.edge_index[0] ,data.batch  
       # print(batch.shape)
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        x = self.conv3(x, edge_index)
        x = self.conv4(x, edge_index)
        x = self.conv5(x, edge_index)


        x = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1)

        x = self.lin1(x)
        x = self.act1(x)
        x = self.lin2(x)
        x = self.act2(x)      
        x = F.dropout(x, p=0.5, training=self.training)

        x = torch.sigmoid(self.lin3(x)).squeeze(1)

        return x