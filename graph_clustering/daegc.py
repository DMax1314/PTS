import argparse
import numpy as np

from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam, AdamW

import utils
from gat_model import GAT


class DAEGC(nn.Module):
    def __init__(self, num_features, hidden_size, embedding_size, alpha, num_clusters, v=1):
        super(DAEGC, self).__init__()
        self.num_clusters = num_clusters
        self.v = v

        # get pretrain model
        self.gat = GAT(num_features, hidden_size, embedding_size, alpha)
        # self.gat.load_state_dict(torch.load(args.pretrain_path, map_location='cpu'))

        # cluster layer
        self.cluster_layer = Parameter(torch.Tensor(num_clusters, embedding_size))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

    def forward(self, x, adj, M):
        A_pred, z = self.gat(x, adj, M)
        q = self.get_Q(z)

        return A_pred, z, q

    def get_Q(self, z):
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        return q


def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


def trainer(dataset):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DAEGC(num_features=2, hidden_size=256,
                  embedding_size=16, alpha=0.2, num_clusters=100).to(device)
    print(model)
    optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=5e-3)

    # data process
    dataset = utils.data_preprocessing(dataset)
    adj = dataset.adj.to(device)
    adj_label = dataset.adj_label.to(device)
    M = utils.get_M(adj).to(device)

    # data and label
    data = torch.Tensor(dataset.x).to(device)
    # y = dataset.y.cpu().numpy()

    with torch.no_grad():
        _, z = model.gat(data, adj, M)

    # get kmeans and pretrain cluster result
    # kmeans = KMeans(n_clusters=args.n_clusters, n_init=20)
    # y_pred = kmeans.fit_predict(z.data.cpu().numpy())
    # model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)
    # eva(y, y_pred, 'pretrain')

    for epoch in range(500):
        model.train()
        if epoch % 10 == 0:
            # update_interval
            A_pred, z, Q = model(data, adj, M)

            # q = Q.detach().data.cpu().numpy().argmax(1)  # Q
            # eva(y, q, epoch)

        A_pred, z, q = model(data, adj, M)
        p = target_distribution(Q.detach())

        kl_loss = F.kl_div(q.log(), p, reduction='batchmean')
        re_loss = F.binary_cross_entropy(A_pred.view(-1), adj_label.view(-1))

        loss = 10 * kl_loss + re_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    A_pred, z, q = model(data, adj, M)
    return z


if __name__ == "__main__":
    dataset = utils.get_dataset()
    outs = trainer(dataset)
