import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv, SAGEConv, APPNPConv, GATConv
from vq import VectorQuantize
from VQ import VectorQuantizer,DataDependentVectorQuantizer
import dgl
from sklearn import metrics
import numpy as np
class MLP(nn.Module):
    def __init__(
        self,
        num_layers,
        input_dim,
        hidden_dim,
        output_dim,
        dropout_ratio,
        norm_type="none",
    ):
        super(MLP, self).__init__()
        self.num_layers = num_layers
        self.norm_type = norm_type
        self.dropout = nn.Dropout(dropout_ratio)
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.linear = nn.Linear(hidden_dim, input_dim)
        if num_layers == 1:
            self.layers.append(nn.Linear(input_dim, output_dim))
        else:
            self.layers.append(nn.Linear(input_dim, hidden_dim))
            if self.norm_type == "batch":
                self.norms.append(nn.BatchNorm1d(hidden_dim))
            elif self.norm_type == "layer":
                self.norms.append(nn.LayerNorm(hidden_dim))

            for i in range(num_layers - 2):
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
                if self.norm_type == "batch":
                    self.norms.append(nn.BatchNorm1d(hidden_dim))
                elif self.norm_type == "layer":
                    self.norms.append(nn.LayerNorm(hidden_dim))

            self.layers.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, feats):
        h = feats
        h_list = []
        for l, layer in enumerate(self.layers):
            h = layer(h)
            if l != self.num_layers - 1:
                h_list.append(h)
                if self.norm_type != "none":
                    h = self.norms[l](h)
                h = F.relu(h)
                h = self.dropout(h)
                vq = self.linear(h)
                h_list.append(vq)
        return h_list, h


"""
Adapted from the SAGE implementation from the official DGL example
https://github.com/dmlc/dgl/blob/master/examples/pytorch/ogb/ogbn-products/graphsage/main.py
"""
class GCN(nn.Module):
    def __init__(
        self,
        num_layers,
        input_dim,
        hidden_dim,
        output_dim,
        dropout_ratio,
        activation,
        norm_type,
        codebook_size,
        lamb_edge,
        lamb_node
    ):
        super().__init__()
        self.num_layers = num_layers
        self.norm_type = norm_type
        self.dropout = nn.Dropout(dropout_ratio)
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.graph_layer_1 = GraphConv(input_dim, input_dim, activation=activation)
        self.graph_layer_2 = GraphConv(input_dim, hidden_dim, activation=activation)
        self.decoder_1 = nn.Linear(input_dim, input_dim)
        self.decoder_2 = nn.Linear(input_dim, input_dim)
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.vq = VectorQuantize(dim=input_dim, codebook_size=codebook_size, decay=0.8,commitment_weight=0.25, use_cosine_sim = True)
        self.lamb_edge = lamb_edge
        self.lamb_node = lamb_node

    def forward(self, g, feats):
        h = feats
        
        g=g.cpu()
        adj = g.adjacency_matrix().to_dense().to(feats.device)
        g=g.to(feats.device)

        h_list = []
        h = self.graph_layer_1(g, h)
        if self.norm_type != "none":
            h = self.norms[0](h)
        h = self.dropout(h)
        h_list.append(h)
        quantized, _, commit_loss, dist, codebook = self.vq(h)
        
        quantized_edge = self.decoder_1(quantized)
        quantized_node = self.decoder_2(quantized)

        feature_rec_loss = self.lamb_node * F.mse_loss(h, quantized_node)
        adj_quantized = torch.matmul(quantized_edge, quantized_edge.t())
        adj_quantized = (adj_quantized - adj_quantized.min()) / (adj_quantized.max() - adj_quantized.min())
        edge_rec_loss = self.lamb_edge * torch.sqrt(F.mse_loss(adj, adj_quantized))
        

        dist = torch.squeeze(dist)
        h_list.append(quantized)
        h = self.graph_layer_2(g, quantized_edge)
        h_list.append(h)
        h = self.linear(h)
        loss = feature_rec_loss + edge_rec_loss + commit_loss
        
        return h_list, h, loss, dist, codebook



class SAGE(nn.Module):
    def __init__(
        self,
        num_layers,
        input_dim,
        hidden_dim,
        output_dim,
        dropout_ratio,
        activation,
        norm_type,
        codebook_size,
        lamb_edge,
        lamb_node,
    ):
        super().__init__()
        self.num_layers = 3
        self.norm_type = norm_type
        self.dropout = dropout_ratio
        
        #self.node_embedding = nn.Embedding(169343,128)
        
        self.layers = nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        
        self.norms_1=torch.nn.BatchNorm1d(256)
        self.norms_2=torch.nn.BatchNorm1d(256)
        self.norms_3=torch.nn.BatchNorm1d(256)
        self.norms_4=torch.nn.BatchNorm1d(256)
        
        self.input_dim = input_dim
        self.hidden_dim = 256
        self.output_dim = output_dim
        
        self.graph_layer_1 = GraphConv(input_dim, 256, activation=activation)
        self.graph_layer_2 = GraphConv(256, 256, activation=activation)
        
        self.graph_layer_3 = GraphConv(input_dim, 256, activation=activation)
        self.graph_layer_4 = GraphConv(256, 256, activation=activation)
        
        self.proj = nn.Linear(256,codebook_size)
        
        #self.graph_layer_5 = GraphConv(256, 256, activation=activation)
        self.decoder_1 = nn.Linear(256, 128)
        self.decoder_2 = nn.Linear(256, 256)
        self.linear = nn.Linear(4096, output_dim)
        self.codebook_size = codebook_size
        
        #self.vq = VectorQuantize(dim=256, codebook_size=codebook_size, decay=0.9,commitment_weight=0.25, use_cosine_sim = True)
        self.vq = DataDependentVectorQuantizer(dim=256, codebook_size=codebook_size,commitment_weight=0.25)
        self.lamb_edge = lamb_edge
        self.lamb_node = lamb_node
    def forward(self, blocks, feats,ori_g):
        h = feats
        h_list = []
        g = dgl.DGLGraph().to(h.device)
        g.add_nodes(h.shape[0])
        blocks = [blk.int() for blk in blocks]
        for block in blocks:
            src, dst = block.all_edges()
            src = src.type(torch.int64).cuda()
            dst = dst.type(torch.int64).cuda()
            g.add_edges(src,dst)
            g.add_edges(src,src)
            g.add_edges(dst,src)
            g.add_edges(dst,dst)
        # print(g)
        
        adj = ori_g.subgraph(blocks[-1].cpu().dstdata[dgl.NID]).adjacency_matrix().to_dense().cuda()
        n = adj.size(0)
        mask = torch.ones(n, n, dtype=torch.bool)
        mask.fill_diagonal_(0)
        non_self_connected_edges = torch.nonzero(adj * mask.cuda(), as_tuple=True)
        all_involved_nodes = torch.cat((non_self_connected_edges[0], non_self_connected_edges[1]))
        unique_nodes = torch.unique(all_involved_nodes)
        sorted_nodes = torch.sort(unique_nodes).values
        adj=adj[sorted_nodes,:]
        adj=adj[:,sorted_nodes]
        h_list = []
        
        
        h_node = self.graph_layer_1(g, h)
        h_node=self.norms_1(h_node)
        h_node = F.dropout(h_node, p=self.dropout, training=self.training)
        h_list.append(h_node)
        h_node = self.graph_layer_2(g, h_node)
        h_node=self.norms_2(h_node)
        h_node = F.dropout(h_node, p=self.dropout, training=self.training)
        h_list.append(h_node)
        
        
        input_nodes=blocks[0].ndata[dgl.NID]['_N']
        
        #h=self.node_embedding.weight[input_nodes]
        
        h_link = self.graph_layer_3(g, h)
        h_link=self.norms_3(h_link)
        h_link = F.dropout(h_link, p=self.dropout, training=self.training)
        h_list.append(h_link)
        h_link = self.graph_layer_4(g, h_link)
        h_link=self.norms_4(h_link)
        h_link = F.dropout(h_link, p=self.dropout, training=self.training)
        h_list.append(h_link)
        
        h_node=h_node[:blocks[-1].num_dst_nodes()]
        h_link=h_link[:blocks[-1].num_dst_nodes()]
        #h_link=h_link[sorted_nodes]


        h=torch.cat([h_node,h_link],axis=0)
        
        
        #
        #quantized, perplexity, commit_loss, dist, codebook = self.vq(h)
        #
        
        h=self.proj(h)
        commit_loss,quantized,perplexity,dist,codebook=self.vq(h)
        #
        
        
        
        quantized_node = self.decoder_1(quantized[:(h_node.shape[0])])
        
        #feature_rec_loss = 10 * F.mse_loss(feats[:blocks[-1].num_dst_nodes()], quantized_node)
        feature_rec_loss =torch.mul(F.normalize(feats[:blocks[-1].num_dst_nodes()],p=2,dim=-1),F.normalize(quantized_node,p=2,dim=-1)).sum(dim=1)
        feature_rec_loss=10*((1-feature_rec_loss)*(1-feature_rec_loss)).mean()


        
        
        # pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
        # quantized_edge = self.decoder_2(quantized[-(h_link.shape[0]):])[sorted_nodes]
        # adj_quantized = torch.matmul(quantized_edge, quantized_edge.t())
        # adj_quantized = F.sigmoid(adj_quantized)
        # adj_quantized,adj=adj_quantized.view(-1), adj.view(-1)
        # weight_mask = (adj== 1)
        # weight_tensor = torch.ones(weight_mask.size(0))
        # weight_tensor[weight_mask] = pos_weight
        # edge_rec_loss = F.binary_cross_entropy(adj_quantized, adj, reduction='mean',weight=weight_tensor.cuda())
        
        
        quantized_edge = self.decoder_2(quantized[-(h_link.shape[0]):])[sorted_nodes]
        adj_quantized = torch.matmul(quantized_edge, quantized_edge.t())
        adj_quantized = (adj_quantized - adj_quantized.min()) / (adj_quantized.max() - adj_quantized.min())
        edge_rec_loss = 1 * torch.sqrt(F.mse_loss(adj, adj_quantized))
        
        
        #print(metrics.roc_auc_score(adj.cpu().detach().numpy(), adj_quantized.cpu().detach().numpy()))
        
        dist = torch.squeeze(dist)
        h_list.append(quantized)
        
        #h = self.graph_layer_5(g, quantized[:(h_node.shape[0])])
        #h_list.append(h)
        
        h = self.linear(h)
        
        loss = feature_rec_loss + edge_rec_loss + commit_loss
        print('perplexity')
        print(perplexity)
        print('feature_rec_loss')
        print(feature_rec_loss)
        print('edge_rec_loss')
        print(edge_rec_loss)
        print('commit_loss')
        print(commit_loss)

        h = h[:blocks[-1].num_dst_nodes()]

        return h_list, h, loss, dist, codebook,perplexity,quantized

    def inference(self, dataloader, feats,g):
        """
        Inference with the GraphSAGE model on full neighbors (i.e. without neighbor sampling).
        dataloader : The entire graph loaded in blocks with full neighbors for each node.
        feats : The input feats of entire node set.
        """
        with torch.no_grad():
            device = feats.device
            dist_node = torch.zeros(feats.shape[0],self.codebook_size, device=device)
            dist_link = torch.zeros(feats.shape[0],self.codebook_size, device=device)
            gumbel_emb_node = torch.zeros(feats.shape[0],256, device=device)
            gumbel_emb_link = torch.zeros(feats.shape[0],256, device=device)
            y = torch.zeros(feats.shape[0], self.output_dim, device=device)
            total_loss=0
            total_perplexity=0
            unique_ind=np.array([])
            #print(y.shape)
            for input_nodes, output_nodes, blocks in dataloader:
                batch_feats = feats[input_nodes]
                h_list, h, loss, dist, codebook,perplexity,quantized=self.forward(blocks,batch_feats,g)
                min_encoding_indices = torch.argmin(dist, dim=1)
                unique_ind=np.concatenate((unique_ind,min_encoding_indices.cpu().detach().numpy()))
                #print(unique_ind)
                total_loss+=loss.item()
                total_perplexity+=perplexity.item()
                dist_node[output_nodes] = dist[:int(dist.shape[0]/2)].detach()
                dist_link[output_nodes] = dist[int(-dist.shape[0]/2):].detach()
                
                gumbel_emb_node[output_nodes] = quantized[:int(quantized.shape[0]/2)].detach()
                gumbel_emb_link[output_nodes] = quantized[int(-quantized.shape[0]/2):].detach()
                
                y[output_nodes] = h.detach()
            #min_indices = torch.argmin(dist_node, dim=1)
            #print(dist_node)
            #print(min_indices)
            
            #unique_columns, _ = torch.unique(min_indices, return_counts=True)
            print("used",len(np.unique(unique_ind)))
            

        return h_list, y, total_loss/len(dataloader), dist_node,dist_link, codebook,total_perplexity,gumbel_emb_node,gumbel_emb_link
        
        
class GAT(nn.Module):
    def __init__(
        self,
        num_layers,
        input_dim,
        hidden_dim,
        output_dim,
        dropout_ratio,
        activation,
        num_heads=8,
        attn_drop=0.3,
        negative_slope=0.2,
        residual=False,
    ):
        super(GAT, self).__init__()
        # For GAT, the number of layers is required to be > 1
        assert num_layers > 1

        hidden_dim //= num_heads
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        self.activation = activation

        heads = ([num_heads] * num_layers) + [1]
        # input (no residual)
        self.layers.append(
            GATConv(
                input_dim,
                hidden_dim,
                heads[0],
                dropout_ratio,
                attn_drop,
                negative_slope,
                False,
                self.activation,
            )
        )

        for l in range(1, num_layers - 1):
            # due to multi-head, the in_dim = hidden_dim * num_heads
            self.layers.append(
                GATConv(
                    hidden_dim * heads[l - 1],
                    hidden_dim,
                    heads[l],
                    dropout_ratio,
                    attn_drop,
                    negative_slope,
                    residual,
                    self.activation,
                )
            )

        self.layers.append(
            GATConv(
                hidden_dim * heads[-2],
                output_dim,
                heads[-1],
                dropout_ratio,
                attn_drop,
                negative_slope,
                residual,
                None,
            )
        )

    def forward(self, g, feats):
        h = feats
        h_list = []
        for l, layer in enumerate(self.layers):
            # [num_head, node_num, nclass] -> [num_head, node_num*nclass]
            h = layer(g, h)
            if l != self.num_layers - 1:
                h = h.flatten(1)
                h_list.append(h)
            else:
                h = h.mean(1)
        return h_list, h


class APPNP(nn.Module):
    def __init__(
        self,
        num_layers,
        input_dim,
        hidden_dim,
        output_dim,
        dropout_ratio,
        activation,
        norm_type="none",
        edge_drop=0.5,
        alpha=0.1,
        k=10,
    ):

        super(APPNP, self).__init__()
        self.num_layers = num_layers
        self.norm_type = norm_type
        self.activation = activation
        self.dropout = nn.Dropout(dropout_ratio)
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        if num_layers == 1:
            self.layers.append(nn.Linear(input_dim, output_dim))
        else:
            self.layers.append(nn.Linear(input_dim, hidden_dim))
            if self.norm_type == "batch":
                self.norms.append(nn.BatchNorm1d(hidden_dim))
            elif self.norm_type == "layer":
                self.norms.append(nn.LayerNorm(hidden_dim))

            for i in range(num_layers - 2):
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
                if self.norm_type == "batch":
                    self.norms.append(nn.BatchNorm1d(hidden_dim))
                elif self.norm_type == "layer":
                    self.norms.append(nn.LayerNorm(hidden_dim))

            self.layers.append(nn.Linear(hidden_dim, output_dim))

        self.propagate = APPNPConv(k, alpha, edge_drop)
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, g, feats):
        h = feats
        h_list = []
        for l, layer in enumerate(self.layers):
            h = layer(h)

            if l != self.num_layers - 1:
                h_list.append(h)
                if self.norm_type != "none":
                    h = self.norms[l](h)
                h = self.activation(h)
                h = self.dropout(h)

        h = self.propagate(g, h)
        return h_list, h


class Model(nn.Module):
    """
    Wrapper of different models
    """

    def __init__(self, conf):
        super(Model, self).__init__()
        self.model_name = conf["model_name"]
        if "MLP" in conf["model_name"]:
            self.encoder = MLP(
                num_layers=conf["num_layers"],
                input_dim=conf["feat_dim"],
                hidden_dim=conf["hidden_dim"],
                output_dim=conf["label_dim"],
                dropout_ratio=conf["dropout_ratio"],
                norm_type=conf["norm_type"],
            ).to(conf["device"])
        elif "SAGE" in conf["model_name"]:
            self.encoder = SAGE(
                num_layers=conf["num_layers"],
                input_dim=conf["feat_dim"],
                hidden_dim=conf["hidden_dim"],
                output_dim=conf["label_dim"],
                dropout_ratio=conf["dropout_ratio"],
                activation=F.relu,
                norm_type=conf["norm_type"],
                codebook_size=conf["codebook_size"],
                lamb_edge=conf["lamb_edge"],
                lamb_node=conf["lamb_node"]
            ).to(conf["device"])
        elif "GCN" in conf["model_name"]:
            self.encoder = GCN(
                num_layers=conf["num_layers"],
                input_dim=conf["feat_dim"],
                hidden_dim=conf["hidden_dim"],
                output_dim=conf["label_dim"],
                dropout_ratio=conf["dropout_ratio"],
                activation=F.relu,
                norm_type=conf["norm_type"],
                codebook_size=conf["codebook_size"],
                lamb_edge=conf["lamb_edge"],
                lamb_node=conf["lamb_node"]
            ).to(conf["device"])
        elif "GAT" in conf["model_name"]:
            self.encoder = GAT(
                num_layers=conf["num_layers"],
                input_dim=conf["feat_dim"],
                hidden_dim=conf["hidden_dim"],
                output_dim=conf["label_dim"],
                dropout_ratio=conf["dropout_ratio"],
                activation=F.relu,
                attn_drop=conf["attn_dropout_ratio"],
            ).to(conf["device"])
        elif "APPNP" in conf["model_name"]:
            self.encoder = APPNP(
                num_layers=conf["num_layers"],
                input_dim=conf["feat_dim"],
                hidden_dim=conf["hidden_dim"],
                output_dim=conf["label_dim"],
                dropout_ratio=conf["dropout_ratio"],
                activation=F.relu,
                norm_type=conf["norm_type"],
            ).to(conf["device"])

    def forward(self, data, feats,g):
        """
        data: a graph `g` or a `dataloader` of blocks
        """
        if "MLP" in self.model_name:
            return self.encoder(feats)
        else:
            return self.encoder(data, feats,g)

    def forward_fitnet(self, data, feats):
        """
        Return a tuple (h_list, h)
        h_list: intermediate hidden representation
        h: final output
        """
        if "MLP" in self.model_name:
            return self.encoder(feats)
        else:
            return self.encoder(data, feats)

    def inference(self, data, feats,g):
        if "SAGE" in self.model_name:
            # return self.forward(data, feats)

            return self.encoder.inference(data, feats,g)
        else:
            return self.forward(data, feats)
