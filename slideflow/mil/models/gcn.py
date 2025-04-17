import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
#import pytorch_lightning as pl
from torch_geometric.nn import GCNConv, global_mean_pool,dense_mincut_pool
from torch.nn import BatchNorm1d
from slideflow.model.torch_utils import get_device

from torch.optim.lr_scheduler import OneCycleLR
from torch_geometric.data import DataLoader
from einops import rearrange
from slideflow.mil.models.vit import VisionTransformer, default_cfgs#, load_pretrained_weights 
from torch_geometric.utils import to_dense_adj 
class GCNBlock(nn.Module):
    def __init__(self, input_dim, output_dim, bn=0, add_self=0, normalize_embedding=0, dropout=0.0, relu=0, bias=True):
        super(GCNBlock, self).__init__()
        self.add_self = add_self
        self.dropout = dropout
        self.relu = relu
        self.bn = bn
        if dropout > 0.001:
            self.dropout_layer = nn.Dropout(p=dropout)
        if self.bn:
            self.bn_layer = nn.BatchNorm1d(output_dim)

        self.normalize_embedding = normalize_embedding
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        torch.nn.init.xavier_normal_(self.weight)
        if bias:
            self.bias = nn.Parameter(torch.zeros(output_dim))
        else:
            self.bias = None

    def forward(self, x, edge_index):
        # Convert edge_index to adjacency matrix
        adj = torch_geometric.utils.to_dense_adj(edge_index).squeeze(0)
        
        # Apply GCN operations
        y = torch.matmul(adj, x)
        if self.add_self:
            y += x
        y = torch.matmul(y, self.weight)
        if self.bias is not None:
            y = y + self.bias
        if self.normalize_embedding:
            y = F.normalize(y, p=2, dim=1)
        if self.bn:
            y = self.bn_layer(y)
        if self.dropout > 0.001:
            y = self.dropout_layer(y)
        if self.relu == 'relu':
            y = F.relu(y)
        elif self.relu == 'lrelu':
            y = F.leaky_relu(y, 0.1)
        return  y


class GCN(nn.Module):
    def __init__(self, n_in,  n_out, hidden_channels = 64,**kwargs):
        super().__init__()
        self.embed_dim = 64
        self.num_layers = 3
        self.node_cluster_num = 100
        
        self.transformer = VisionTransformer(num_classes=n_out, embed_dim=self.embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.bn = 1
        self.add_self = 1
        self.normalize_embedding = 1
   
        self.gcn2 = GCNBlock(n_in,self.embed_dim,self.bn,self.add_self,self.normalize_embedding,0.,0) 
    

    def forward(self, x, edge_index = None, batch = None, graphcam_flag=False):
        # Forward pass through GCN layers
        print(x)
        
   
        
        adj = torch_geometric.utils.to_dense_adj(edge_index).squeeze(0)
        # print("adj",adj)
        # print("adj",x.shape)
        #     # print("edge_index",edge_index)
        # print("edge_index shp",edge_index.shape)
        # print("adj",adj.shape)
        # print("adj",x.shape[0])
        #     # print("edge_index",edge_index)
        # print("edge_index shp",edge_index.shape[0])
        # print("adj",adj.shape[0])
        if(x.shape[0]!=adj.shape[0]):
            
            print("adj",x.shape)
            # print("edge_index",edge_index)
            print("edge_index shp",edge_index.shape)
            print("adj",adj.shape)
            exit()
        cls_loss = x.new_zeros(self.num_layers)
        rank_loss = x.new_zeros(self.num_layers - 1)
        x = self.gcn2(x, edge_index)
       
        
        x = global_mean_pool(x, batch)
        b, _ = x.shape
      
        cls_token = self.cls_token.repeat(b, 1, 1)
        x = torch.cat([cls_token, x.unsqueeze(1)], dim=1)

        x = self.transformer(x)

        
        return x  #, mc1,o1
    
        # x = global_mean_pool(x, batch)
        # print("sshhhaapppeee11",x.shape)
        # x = self.fc(x)
        # print("sshhhaapppeee",x.shape)
        
        # print("shappppeeess",x.shape)
        
        # loss = self.criterion(x, label)
        
        # loss = loss + mc1 + o1
        # pred = x.data.max(1)[1]
        
        # # Global Pooling
        # x = global_mean_pool(x, batch)  # Resulting shape is [batch_size, 96]
        # cls_token = self.cls_token.expand(x.size(0), -1)  # Resulting shape is [batch_size, 96]
        # x = torch.cat((cls_token, x), dim=1)  # Combine them into shape [batch_size, 192]

        # # Now you need to adjust the dimensions of x to match the input expected by self.fc
        # x = x.view(batch_size, -1)  # Reshape it if necessary (e.g., [batch_size, 32])

        # x = self.fc(x)
    # --- FastAI compatibility -------------------------------------------------

    def relocate(self):
        """Move model to GPU. Required for FastAI compatibility."""
        device = get_device()
        self.to(device)

    def plot(*args, **kwargs):
        """Override to disable FastAI plotting."""
        pass


