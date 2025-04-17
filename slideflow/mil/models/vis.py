import torch.nn as nn
import torch
from einops import rearrange
from huggingface_hub import PyTorchModelHubMixin


class SummaryMixing(nn.Module):
    def __init__(self, input_dim, dimensions_f, dimensions_s, dimensions_c):
        super().__init__()
        
        self.local_norm = nn.LayerNorm(dimensions_f)
        self.summary_norm = nn.LayerNorm(dimensions_s)

        self.s = nn.Linear(input_dim, dimensions_s)
        self.f = nn.Linear(input_dim, dimensions_f)
        self.c = nn.Linear(dimensions_s + dimensions_f, dimensions_c)

    def forward(self, x):
        local_summ = torch.nn.GELU()(self.local_norm(self.f(x)))
        time_summ = self.s(x)    
        time_summ = torch.nn.GELU()(self.summary_norm(torch.mean(time_summ, dim=1)))
        time_summ = time_summ.unsqueeze(1).repeat(1, x.shape[1], 1)
        out = torch.nn.GELU()(self.c(torch.cat([local_summ, time_summ], dim=-1)))

        return out


class MultiHeadSummary(nn.Module):
    def __init__(self, nheads, input_dim, dimensions_f, dimensions_s, dimensions_c, dimensions_projection):
        super().__init__()

        self.mixers = nn.ModuleList([SummaryMixing(input_dim=input_dim, dimensions_f=dimensions_f, dimensions_s=dimensions_s, dimensions_c=dimensions_c) for _ in range(nheads)])
        self.projection = nn.Linear(nheads * dimensions_c, dimensions_projection)

    def forward(self, x):
        outs = [mixer(x) for mixer in self.mixers]
        outs = torch.cat(outs, dim=-1)
        return self.projection(outs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


class SummaryTransformer(nn.Module):
    def __init__(self, input_dim, depth, nheads, dimensions_f, dimensions_s, dimensions_c):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleList([MultiHeadSummary(nheads, input_dim, dimensions_f, dimensions_s, dimensions_c, input_dim), FeedForward(input_dim, input_dim)])
            for _ in range(depth)
        ])

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


# class vis(nn.Module, PyTorchModelHubMixin):
#     def __init__(self,  input_dim,num_classes, depth=6, nheads=16,
#                  dimensions_f=64, dimensions_s=64, dimensions_c=64,
#                  num_clusters=512, device='cuda:0'):
#         super().__init__()

#         self.pos_emb1D = nn.Parameter(torch.randn(1,num_clusters, input_dim))
#         self.transformer = SummaryTransformer(input_dim, depth, nheads, dimensions_f, dimensions_s, dimensions_c)
#         self.to_latent = nn.Identity()
#         self.linear_head = nn.Sequential(
#             nn.LayerNorm(input_dim),
#             nn.Linear(input_dim, num_classes)
#         )
#         self.device = device
        
#         self.num_classes1=num_classes
#         self.num_clusters=num_clusters
  
#     def forward(self, x):
#         # print(x.shape)
#         # exit()
#         x = rearrange(x, 'b ... d -> b (...) d')
#         print(x.shape)
#         print(self.pos_emb1D[:, :x.shape[1], :].shape)
#         exit()
#         # if x.shape[1]>512:
#         #     print(x.shape)
#         #     exit()
#         x = x + #self.pos_emb1D.unsqueeze(0)
#         # print(x.shape)
#         # exit()
#         # x = rearrange(x, 'b ... d -> b (...) d') + self.pos_emb1D
#         x = self.transformer(x)
#         x = x.mean(dim=1)
#         x = self.to_latent(x)
#         return self.linear_head(x)
class vis(nn.Module, PyTorchModelHubMixin):
    def __init__(self, input_dim, num_classes, depth=6, nheads=16,
                 dimensions_f=64, dimensions_s=64, dimensions_c=64,
                 num_clusters=512, device='cuda:0'):
        super().__init__()

        self.num_clusters = num_clusters
        self.device = device
        
        # Define positional embedding, ensuring it covers num_clusters
        self.pos_emb1D = nn.Parameter(torch.randn(1, num_clusters, input_dim))

        # Define transformer
        self.transformer = SummaryTransformer(input_dim, depth, nheads, dimensions_f, dimensions_s, dimensions_c)

        # Linear head for classification
        self.to_latent = nn.Identity()
        self.linear_head = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, num_classes)
        )

    def forward(self, x):
    
        x = x.view(x.shape[0], -1, x.shape[-1])  # Explicit reshape
   
        if x.shape[1] > self.num_clusters:
            x = x[:, :self.num_clusters, :]  # Truncate
        elif x.shape[1] < self.num_clusters:

            pad_size = self.num_clusters - x.shape[1]
            x = torch.cat([x, torch.zeros(x.shape[0], pad_size, x.shape[2], device=x.device)], dim=1)


        x = x + self.pos_emb1D[:, :x.shape[1], :]

        x = self.transformer(x)
  
        x = x.mean(dim=1)

        x = self.to_latent(x)
        x = self.linear_head(x)
   
        return x
