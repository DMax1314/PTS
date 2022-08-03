import numpy as np
import networkx as nx
from math import radians, cos, sin, asin, sqrt
import collections
import community
from community import community_louvain
from datetime import datetime
from copy import deepcopy
from deepsnap.graph import Graph
from deepsnap.dataset import GraphDataset
from deepsnap.batch import Batch
from torch_geometric.data import InMemoryDataset, Data
import torch
from torch.utils.data import DataLoader

data = "cluster_data_250.csv"
with open(data, 'r') as f:
    datas = f.readlines()
    datas = datas[1:]
    datas_new = list(map(lambda x:x.strip().split(','), datas))
    datas_new = np.array(datas_new)

np.set_printoptions(precision = 16)
pickup_long = np.array(datas_new[:,5], dtype=float)
pickup_lat = np.array(datas_new[:,6], dtype=float)
dropout_long = np.array(datas_new[:,9], dtype=float)
dropout_lat = np.array(datas_new[:,10], dtype=float)


def clean(pickup_long, pickup_lat):
    pickup_long_index1 = set(np.where(pickup_long > -75)[0])
    pickup_long_index2 = set(np.where(pickup_long < -73)[0])
    pickup_long_index = pickup_long_index1.intersection(pickup_long_index2)

    pickup_lat_index1 = set(np.where(pickup_lat > 40)[0])
    pickup_lat_index2 = set(np.where(pickup_lat < 42)[0])
    pickup_lat_index = pickup_lat_index1.intersection(pickup_lat_index2)

    pickup_index = pickup_long_index.intersection(pickup_lat_index)
    return pickup_index

pickup_index = clean(pickup_long, pickup_lat)
dropout_index = clean(dropout_long, dropout_lat)
final_index = np.array(list(pickup_index.intersection(dropout_index)))

pickup = np.vstack((pickup_long, pickup_lat))
pickup_final = pickup[:, final_index]
dropout = np.vstack((dropout_long, dropout_lat))
dropout_final = dropout[:, final_index]

min_long = min(np.min(pickup_final[0,:]), np.min(dropout_final[0,:]))
max_lat = max(np.max(pickup_final[1,:]), np.max(dropout_final[1,:]))
max_long = max(np.max(pickup_final[0,:]), np.max(dropout_final[0,:]))
min_lat = min(np.min(pickup_final[1,:]), np.min(dropout_final[1,:]))

t_pick = datas_new[:,1][final_index]
t_drop = datas_new[:,2][final_index]

def turn_format(t_pick):
    tms = []
    for i in range(t_pick.shape[0]):
        temp = t_pick[i].strip().split()
        temp1 = list(map(int, temp[0].split('-')))
        temp2 = list(map(int, temp[1].split(':')))
        temp1.extend(temp2)
        timec = datetime(temp1[0], temp1[1], temp1[2], temp1[3], temp1[4], temp1[5])
        tms.append(timec)
    return np.array(tms)

pickup_timestamp = turn_format(t_pick)
dropoff_timestamp = turn_format(t_drop)
ptime_min = np.min(pickup_timestamp)
dtime_min = np.min(dropoff_timestamp)
ptime_max = np.max(pickup_timestamp)
dtime_max = np.max(dropoff_timestamp)
time_min = min(ptime_min, dtime_min)
time_max = max(ptime_max, dtime_max)
print(time_min)
print(time_max)
time_gap = 240*60
time_num = int((datetime.timestamp(time_max)-datetime.timestamp(time_min))//time_gap)  # 4h
print(time_num)

def geodistance(lng1,lat1,lng2,lat2):
    lng1, lat1, lng2, lat2 = map(radians, [float(lng1), float(lat1), float(lng2), float(lat2)])
    dlon=lng2-lng1
    dlat=lat2-lat1
    a=sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    distance=2*asin(sqrt(a))*6371*1000
    distance=round(distance/1000,3)
    return distance

lat_length = geodistance(min_long, min_lat, min_long, max_lat)*1000
lng_length = geodistance(min_long, min_lat, max_long, min_lat)*1000
grid_length = 50 #50m
rows = int(lat_length/grid_length)+1
cols = int(lng_length/grid_length)+1
pdr_dict = collections.defaultdict(list)
ptime_dict = dict()
dtime_dict = dict()
lat_matrix = np.zeros((rows, cols))
lng_matrix = np.zeros((rows, cols))
print("shape:",lat_matrix.shape)

long_gap = float(abs(min_long-max_long)/cols)
lat_gap = float(abs(min_lat-max_lat)/rows)

def vals(pickup_final, ptime_dict, ptimestamp):
    pickup_index1 = np.array((pickup_final[0,:]-min_long)/long_gap, dtype=int)
    pickup_index1 = np.clip(pickup_index1, 0, cols-1)
    pickup_index2 = np.array((max_lat-pickup_final[1,:])/lat_gap, dtype=int)
    pickup_index2 = np.clip(pickup_index2, 0, rows-1)
    for i in range(pickup_final.shape[1]):
        pdr_dict[(pickup_index2[i], pickup_index1[i])].append(pickup_final[:,i])
        if (pickup_index2[i], pickup_index1[i]) not in ptime_dict:
            ptime_dict[(pickup_index2[i], pickup_index1[i])] = np.zeros(time_num+1)
            index = int((datetime.timestamp(ptimestamp[i])-datetime.timestamp(time_min))//time_gap)
            ptime_dict[(pickup_index2[i], pickup_index1[i])][index]+=1
        else:
            index = int((datetime.timestamp(ptimestamp[i]) - datetime.timestamp(time_min)) // time_gap)
            ptime_dict[(pickup_index2[i], pickup_index1[i])][index] += 1

vals(pickup_final, ptime_dict, pickup_timestamp)
vals(dropout_final, dtime_dict, dropoff_timestamp)
time_dict= deepcopy(ptime_dict)
for key in dtime_dict:
    if key in time_dict:
        time_dict[key]+=dtime_dict[key]
    else:
        time_dict[key]=dtime_dict[key]
for key in time_dict:
    print(time_dict[key])
    break

sums = 0
for key in pdr_dict:
    sums+=len(pdr_dict[key])
avg = sums/len(pdr_dict.keys())
print("avg:", avg)

for key in pdr_dict:
    if len(pdr_dict[key])<avg:
        continue
    tmp = np.array(pdr_dict[key])
    lat_tmp = np.mean(tmp[:,1])
    lng_tmp = np.mean(tmp[:,0])
    lat_matrix[key[0],key[1]] = lat_tmp
    lng_matrix[key[0],key[1]] = lng_tmp

nonzero_num = np.count_nonzero(lat_matrix)
nodes_tmp = list(range(nonzero_num))
nodes = list(map(str, nodes_tmp))
nonzero_dict = dict()
coor_dict = dict()
pos_dict = dict()
id = 0
G = nx.Graph()
for i in range(lat_matrix.shape[0]):
    for j in range(lat_matrix.shape[1]):
        if lat_matrix[i,j]!=0 and lng_matrix[i,j]!=0:
            nonzero_dict[(i,j)] = nodes[id]
            coor_dict[nodes[id]] = time_dict[(i,j)]
            pos_dict[nodes[id]] = (float(lng_matrix[i, j]), float(lat_matrix[i, j]))
            id+=1

edges = []
directions = [[0,1],[1,0],[-1,0],[0,-1],[1,1],[1,-1],[-1,1],[-1,-1]]
rdis = 70
for i in range(lat_matrix.shape[0]):
    for j in range(lat_matrix.shape[1]):
        if lat_matrix[i,j]!=0 and lng_matrix[i,j]!=0:
            for k in range(4):
                i_new = i+directions[k][0]
                j_new = j+directions[k][1]
                if i_new>=0 and j_new>=0 and i_new<lat_matrix.shape[0] and j_new<lat_matrix.shape[1] and lat_matrix[i_new,j_new]!=0 and lng_matrix[i_new,j_new]!=0:
                    dis = geodistance(lng_matrix[i,j],lat_matrix[i,j],lng_matrix[i_new, j_new],lat_matrix[i_new,j_new])
                    if dis*1000<=rdis:
                        edges.append((nonzero_dict[(i,j)],nonzero_dict[(i_new,j_new)]))

G.add_nodes_from(nodes)
G.add_edges_from(edges)
# remove = [node for node,degree in G.degree() if degree == 0]
remove = []
for node in G.nodes():
    if G.degree(node)==0:
        remove.append(node)
G.remove_nodes_from(remove)
print("node number:", G.number_of_nodes())
print("edge number:", G.number_of_edges())

def preprocess(G, method="louvain"):
    nodes_new = []
    node_feature = {node: torch.tensor(list(coor_dict[node])) for node in G.nodes()}
    nx.set_node_attributes(G, node_feature, name='node_feature')
    graphs = []
    if method == "louvain":
        community_mapping = community_louvain.best_partition(G, resolution=10)
        communities = {}
        for node in community_mapping:
            comm = community_mapping[node]
            if comm in communities:
                communities[comm].add(node)
            else:
                communities[comm] = set([node])
        communities = communities.values()
    elif method == "bisection":
        communities = nx.algorithms.community.kernighan_lin_bisection(G)
    elif method == "greedy":
        communities = nx.algorithms.community.greedy_modularity_communities(G)

    for community in communities:
        nodes = set(community)
        subgraph = G.subgraph(nodes)
        nodes_new.extend(list(subgraph.nodes))
        # Make sure each subgraph has more than 10 nodes
        if subgraph.number_of_nodes() > 10:
            dg = Graph(subgraph)
            graphs.append(dg)
    return graphs, nodes_new

def build():
    graph, nodes_new = preprocess(G)
    G_new = G.subgraph(nodes_new)
    print(G_new.number_of_nodes())
    print(len(graph))

    '''
    dataset = GraphDataset(graph, task="link_pred",edge_train_mode="disjoint")
    #follow_batch = []
    datasets = {}
    datasets['train'], datasets['val'], datasets['test'] = dataset.split(
                transductive=True, split_ratio=[0.8, 0.1, 0.1])
    dataloader = DataLoader(datasets['train'], collate_fn=Batch.collate(), batch_size=16, shuffle=True)
    '''
    return graph

def record_pos():
    poses = []
    for node in G.nodes():
        poses.append([pos_dict[node][0], pos_dict[node][1]])
    return np.array(poses)

build()
