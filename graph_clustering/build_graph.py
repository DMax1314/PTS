import numpy as np
import networkx as nx
from math import radians, cos, sin, asin, sqrt
import collections
import community
from community import community_louvain

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
grid_length = 300 #300m
rows = int(lat_length/grid_length)+1
cols = int(lng_length/grid_length)+1
pdr_dict = collections.defaultdict(list)
lat_matrix = np.zeros((rows, cols))
lng_matrix = np.zeros((rows, cols))
print("shape:",lat_matrix.shape)

long_gap = float(abs(min_long-max_long)/cols)
lat_gap = float(abs(min_lat-max_lat)/rows)

def vals(pickup_final):
    pickup_index1 = np.array((pickup_final[0,:]-min_long)/long_gap, dtype=int)
    pickup_index1 = np.clip(pickup_index1, 0, cols-1)
    pickup_index2 = np.array((max_lat-pickup_final[1,:])/lat_gap, dtype=int)
    pickup_index2 = np.clip(pickup_index2, 0, rows-1)
    for i in range(pickup_final.shape[1]):
        pdr_dict[(pickup_index2[i], pickup_index1[i])].append(pickup_final[:,i])

vals(pickup_final)
vals(dropout_final)
for key in pdr_dict:
    tmp = np.array(pdr_dict[key])
    lat_tmp = np.mean(tmp[:,1])
    lng_tmp = np.mean(tmp[:,0])
    lat_matrix[key[0],key[1]] = lat_tmp
    lng_matrix[key[0],key[1]] = lng_tmp

nonzero_num = np.count_nonzero(lat_matrix)
nodes_tmp = list(range(nonzero_num))
nodes = list(map(str, nodes_tmp))
nonzero_dict = dict()
id = 0
G = nx.Graph()
for i in range(lat_matrix.shape[0]):
    for j in range(lat_matrix.shape[1]):
        if lat_matrix[i,j]!=0 and lng_matrix[i,j]!=0:
            nonzero_dict[(i,j)] = nodes[id]
            id+=1

edges = []
directions = [[0,1],[1,0],[-1,0],[0,-1],[1,1],[1,-1],[-1,1],[-1,-1]]
rdis = 500
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

part = community_louvain.best_partition(G)

print(len(set(part.values())))
