from __future__ import print_function, division
import argparse
import random
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader
from torch.nn import Linear
import utils
from build_graph import record_pos
import matplotlib.pyplot as plt
import plot_map

plot_map.set_mapboxtoken(r'pk.eyJ1IjoibHl1ZCIsImEiOiJjbDVhcjJ6Z3QwMGVwM2lxOGc1dGF0bmlmIn0.-7ibyu3eBAyD8EhBq_2h7g')
plot_map.set_imgsavepath(r'/home/s3084361/data/graph_clustering')
#plot_map.set_imgsavepath(r'/Users/lyudonghang/Downloads/NYDATA/NYTaxi')

class AE(nn.Module):

    def __init__(self, n_enc_1, n_enc_2, n_dec_1, n_dec_2,
                 n_input, n_z):
        super(AE, self).__init__()
        self.enc_1 = Linear(n_input, n_enc_1)
        self.enc_2 = Linear(n_enc_1, n_enc_2)
        self.z_layer = Linear(n_enc_2, n_z)

        self.dec_1 = Linear(n_z, n_dec_1)
        self.dec_2 = Linear(n_dec_1, n_dec_2)
        self.x_bar_layer = Linear(n_dec_2, n_input)

    def forward(self, x):
        enc_h1 = F.relu(self.enc_1(x))
        enc_h2 = F.relu(self.enc_2(enc_h1))
        z = self.z_layer(enc_h2)

        dec_h1 = F.relu(self.dec_1(z))
        dec_h2 = F.relu(self.dec_2(dec_h1))
        x_bar = self.x_bar_layer(dec_h2)

        return x_bar, enc_h1, enc_h2, z

class GATLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, alpha=0.2):
        super(GATLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.a_self = nn.Parameter(torch.zeros(size=(out_features, 1)))
        nn.init.xavier_uniform_(self.a_self.data, gain=1.414)

        self.a_neighs = nn.Parameter(torch.zeros(size=(out_features, 1)))
        nn.init.xavier_uniform_(self.a_neighs.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj, M, concat=True):
        h = torch.mm(input, self.W)

        attn_for_self = torch.mm(h, self.a_self)  # (N,1)
        attn_for_neighs = torch.mm(h, self.a_neighs)  # (N,1)
        attn_dense = attn_for_self + torch.transpose(attn_for_neighs, 0, 1)
        attn_dense = torch.mul(attn_dense, M)
        attn_dense = self.leakyrelu(attn_dense)  # (N,N)

        zero_vec = -9e15 * torch.ones_like(adj)
        adj = torch.where(adj > 0, attn_dense, zero_vec)
        attention = F.softmax(adj, dim=1)
        h_prime = torch.matmul(attention, h)

        if concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.in_features)
            + " -> "
            + str(self.out_features)
            + ")"
        )


class SDCN(nn.Module):
    def __init__(self, n_enc_1, n_enc_2, n_dec_1, n_dec_2,
                n_input, n_time, n_z, n_clusters, alpha, v=1):
        super(SDCN, self).__init__()
        self.ae = AE(
            n_enc_1=n_enc_1,
            n_enc_2=n_enc_2,
            n_dec_1=n_dec_1,
            n_dec_2=n_dec_2,
            n_input=n_input,
            n_z=n_z)
        self.ae.load_state_dict(torch.load("dblp1.pkl", map_location='cpu'))
        self.alpha = alpha
        self.gnn_1 = GATLayer(n_time, n_enc_1, alpha)
        self.gnn_2 = GATLayer(n_enc_1, n_enc_2, alpha)
        self.gnn_3 = GATLayer(n_enc_2, n_z, alpha)
        self.gnn_4 = GATLayer(n_z, n_clusters, alpha)

        self.cluster_layer = Parameter(torch.Tensor(n_clusters, n_z))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)
        self.v = v
    
    def forward(self, x, y, adj, M):
        x_bar, tra1, tra2, z = self.ae(x)
        sigma = 0.5
        
        h = self.gnn_1(y, adj, M)
        h = self.gnn_2((1-sigma)*h + sigma*tra1, adj, M)
        h = self.gnn_3((1-sigma)*h + sigma*tra2, adj, M)
        zf = (1-sigma)*h + sigma*z
        h = self.gnn_4((1-sigma)*h + sigma*z, adj, M)
        predict = F.softmax(h, dim=1)
        
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        
        return x_bar, q, predict, zf

def target_distribution(q):
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()

def train_sdcn(dataset):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cluster_num =10
    model = SDCN(128, 256, 256, 128,
                n_input=3,
                n_time=157,
                n_z=64,
                n_clusters=cluster_num,
                alpha=0.2,
                v=1.0).to(device)
    print(model)

    optimizer = Adam(model.parameters(), lr=3e-4)

    # KNN Graph
    dataset = utils.data_preprocessing(dataset)
    adj = dataset.adj.to(device)      
    adj = adj.cuda()
    M = utils.get_M(adj)
    M = M.cuda()

    # cluster parameter initiate
    data = np.loadtxt("dblp.txt", dtype=float)
    data = torch.Tensor(data).to(device)
    with torch.no_grad():
        _, _, _, z = model.ae(data)

    kmeans = KMeans(n_clusters=cluster_num, n_init=20)
    y_pred = kmeans.fit_predict(z.data.cpu().numpy())
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)

    for epoch in range(200):
        if epoch % 1 == 0:
        # update_interval
            _, tmp_q, pred, _ = model(data, dataset.node_feature, adj, M)
            tmp_q = tmp_q.data
            p = target_distribution(tmp_q)

        x_bar, q, pred, _ = model(data, dataset.node_feature, adj, M)

        kl_loss = F.kl_div(q.log(), p, reduction='batchmean')
        ce_loss = F.kl_div(pred.log(), p, reduction='batchmean')
        re_loss = F.mse_loss(x_bar, data)

        loss = 0.1 * kl_loss + 0.01 * ce_loss + re_loss
        print("epoch:",epoch, "loss:",loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        _, _, _, z = model(data, dataset.node_feature, adj, M)

    y_pred_new = kmeans.fit_predict(z.data.cpu().numpy())
    return y_pred_new
        
         
dataset = utils.get_dataset()
y_pred = train_sdcn(dataset)

colors=["blue","red","green","yellow","brown","orange","purple","grey","pink","black","lightblue","blue"]
def plot_points(poses, outs, bounds):
    fig     = plt.figure(1,(8,8),dpi = 100)
    ax      = plt.subplot(111)
    plt.sca(ax)
    fig.tight_layout(rect = (0.05,0.1,1,0.9))
    plot_map.plot_map(plt,bounds,zoom = 12,style = 4)
    for i in range(poses.shape[0]):
        plt.scatter(poses[i,0],poses[i,1],c = colors[outs[i]] ,s=1)
    plt.axis('off')
    plt.xlim(bounds[0],bounds[2])
    plt.ylim(bounds[1],bounds[3])
    plt.savefig("clusters_v2.png")
    plt.show()

poses = record_pos()  
bounds = [-74.0282068, 40.6957492, -73.9196656, 40.8368934] 
plot_points(poses, y_pred, bounds)
    