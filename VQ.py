import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torch import nn, einsum
import math
class VectorQuantizer(nn.Module):
    def __init__(self, dim, codebook_size, commitment_weight):
        super(VectorQuantizer, self).__init__()
        self.n_e = codebook_size
        self.e_dim = dim
        self.beta = commitment_weight

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        
        
    def forward(self, z):
        """
        Inputs the output of the encoder network z and maps it to a discrete 
        one-hot vector that is the index of the closest embedding vector e_j

        z (continuous) -> z_q (discrete)

        z.shape = (batch, channel, height, width)

        quantization pipeline:

            1. get encoder input (B,C,H,W)
            2. flatten input to (B*H*W,C)

        """
        #z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())

        # find closest encodings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(
            min_encoding_indices.shape[0], self.n_e).to(device)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)

        # compute loss for embedding
        loss = self.beta * torch.mean((z_q.detach()-z)**2) + torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # perplexity
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        # reshape back to match original input shape
        #z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return loss, z_q, perplexity, d, self.embedding.weight


class DataDependentVectorQuantizer(nn.Module):
    def __init__(self, dim, codebook_size, commitment_weight, temperature=1,M_init=-1, M_reestim=300000, reservoir_size=100000):
        super(DataDependentVectorQuantizer, self).__init__()
        self.n_e = codebook_size
        self.e_dim = dim
        self.beta = commitment_weight
        self.reservoir_size = reservoir_size
        self.reservoir = []
        self.M_init = M_init
        self.M_reestim = M_reestim
        self.iter_count = 0
        self.is_initialized = False
        self.temperature=temperature
        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        #self.proj=nn.Linear(256,4096)
        
    def update_reservoir(self, z_flattened):
        if len(self.reservoir) < self.reservoir_size:
            self.reservoir.extend(z_flattened.cpu().detach().numpy())
        else:
            prob = len(self.reservoir) / (len(self.reservoir) + 1)
            for sample in z_flattened.cpu().detach().numpy():
                if np.random.rand() < prob:
                    index = np.random.randint(0, len(self.reservoir))
                    self.reservoir[index] = sample

    def initialize_with_kmeans(self):
        data = np.array(self.reservoir)[:self.reservoir_size]
        kmeans = KMeans(n_clusters=self.n_e, init='k-means++').fit(data)
        self.embedding.weight.data = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32).to(self.embedding.weight.device)
        self.is_initialized = True
        print('Have finished initation')

    def reestimate_codebook(self):
        if self.is_initialized:
            print('Start reestimate')
            data = np.array(self.reservoir)[:self.reservoir_size]
            kmeans = KMeans(n_clusters=self.n_e).fit(data)
            self.embedding.weight.data = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32).to(self.embedding.weight.device)
    def cos_anneal(self,e0, e1, t0, t1, e):
        """ ramp from (e0, t0) -> (e1, t1) through a cosine schedule based on e \in [e0, e1] """
        alpha = max(0, min(1, (e - e0) / (e1 - e0))) # what fraction of the way through are we
        alpha = 1.0 - math.cos(alpha * math.pi/2) # warp through cosine
        t = alpha * t1 + (1 - alpha) * t0 # interpolate accordingly
        return t
    def forward(self, z):
        if self.training:
            self.iter_count += 1
        print(self.iter_count)
        if not self.is_initialized:
            if self.iter_count <= self.M_init:
                # Warm-up period, collect data but do not quantize
                self.update_reservoir(z.view(-1, self.e_dim))
                return 0, z, torch.tensor(0), torch.zeros(z.shape[0],4096).cuda(), self.embedding.weight
            elif self.iter_count == self.M_init + 1:
                # Initialize codebook after warm-up
                self.initialize_with_kmeans()
            
        #self.update_reservoir(z.view(-1, self.e_dim))

        if self.M_reestim is not None and self.iter_count % self.M_reestim == 0:
            print('_______________done________________')
            self.reestimate_codebook()

        # Proceed with normal VQ operations
        if False:
            z_flattened = z
            
            # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
            d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
                torch.sum(self.embedding.weight**2, dim=1) - 2 * \
                torch.matmul(z_flattened, self.embedding.weight.t())
    
            # find closest encodings
            min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
            min_encodings = torch.zeros(
                min_encoding_indices.shape[0], self.n_e).to(z.device)
            min_encodings.scatter_(1, min_encoding_indices, 1)
    
            # get quantized latent vectors
            z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)
            if self.training:
                loss = torch.mean((z_q - z.detach()) ** 2)+self.beta * torch.mean((z_q.detach()-z)**2)
            else:
                loss = 0
    
            # preserve gradients
            z_q = z + (z_q - z).detach()
    
            # perplexity
            e_mean = torch.mean(min_encodings, dim=0)
            perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))
        else:
        
            logits=z
            
            if self.training:
                soft_one_hot = F.gumbel_softmax(logits, tau=self.temperature, dim=1, hard=False)
            else:
                soft_one_hot = F.gumbel_softmax(logits, tau=self.temperature, dim=1, hard=False)
            
            z_q = einsum('b n, n d -> b d', soft_one_hot, self.embedding.weight)
            
            qy = F.softmax(logits, dim=1)
            
            if self.training:
                loss = self.cos_anneal(0,4500,0,1e-2,self.iter_count) * torch.sum(qy * torch.log(qy * self.n_e + 1e-10), dim=1).mean()
            else:
                loss=0
            
            ind = soft_one_hot.argmax(dim=1)
            
            min_encodings = F.one_hot(ind, self.n_e).float().reshape(-1, self.n_e)
            
            avg_probs = min_encodings.mean(0)
            
            perplexity = (-(avg_probs * torch.log(avg_probs + 1e-10)).sum()).exp()
            
            cluster_use = torch.sum(avg_probs > 0)
            print('cluster_use is ',cluster_use)
            d=qy

        return loss, z_q, perplexity, d, self.embedding.weight