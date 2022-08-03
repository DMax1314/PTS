import numpy as np
import csv
import os
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from math import radians, cos, sin, asin, sqrt
import collections
import copy
import plot_map
import matplotlib.pyplot as plt
import networkx as nx
from sknetwork.ranking import PageRank
from datetime import datetime
import torch
# from deepsnap.graph import Graph
# from deepsnap.dataset import GraphDataset

plot_map.set_mapboxtoken(r'pk.eyJ1IjoibHl1ZCIsImEiOiJjbDVhcjJ6Z3QwMGVwM2lxOGc1dGF0bmlmIn0.-7ibyu3eBAyD8EhBq_2h7g')
plot_map.set_imgsavepath(r'/Users/lyudonghang/Downloads/NYDATA/NYTaxi')

path = "../POI"
taxi_path = "../cluster_data_250.csv"
np.set_printoptions(precision=16)
long = []
lat = []
for file in os.listdir(path):
    if "POI" in file:
        path_new = os.path.join(path, file)
        times = 0
        with open(path_new, 'r') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                times += 1
                if times == 1:
                    continue
                long.append(row[0])
                lat.append(row[1])

pickup_long = []
pickup_lat = []
dropoff_long = []
dropoff_lat = []
time_pickup = []
time_dropoff = []
with open(taxi_path, 'r') as f:
    reader = csv.reader(f)
    times = 0
    for row in reader:
        times += 1
        if times == 1:
            continue
        time_pickup.append(row[1])
        time_dropoff.append(row[2])
        pickup_long.append(row[5])
        pickup_lat.append(row[6])
        dropoff_long.append(row[9])
        dropoff_lat.append(row[10])

long = np.array(long, dtype=float).reshape(1, -1)
lat = np.array(lat, dtype=float).reshape(1, -1)
poi_pos = np.concatenate((long, lat), axis=0)
poi_pos = np.transpose(poi_pos)
print(poi_pos.shape)

poi_pos_new = StandardScaler().fit_transform(poi_pos)
db = DBSCAN(eps=0.1, min_samples=20).fit(poi_pos_new)
# db = KMeans(n_clusters=10, n_init=25).fit(poi_pos)
labels = db.labels_
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
print(n_clusters_)
pos_true = labels == 0
poi_pos = poi_pos[pos_true]
lat_min = np.min(poi_pos[:, 1])
lat_max = np.max(poi_pos[:, 1])
long_min = np.min(poi_pos[:, 0])
long_max = np.max(poi_pos[:, 0])
print(poi_pos.shape)

pickup_long = np.array(pickup_long, dtype=float).reshape(-1, 1)
pickup_lat = np.array(pickup_lat, dtype=float).reshape(-1, 1)
pickup = np.concatenate((pickup_long, pickup_lat), axis=1)
dropoff_long = np.array(dropoff_long, dtype=float).reshape(-1, 1)
dropoff_lat = np.array(dropoff_lat, dtype=float).reshape(-1, 1)
dropoff = np.concatenate((dropoff_long, dropoff_lat), axis=1)


def clean(pickup_long, pickup_lat):
    pickup_long_index1 = set(np.where(pickup_long >= long_min)[0])
    pickup_long_index2 = set(np.where(pickup_long <= long_max)[0])
    pickup_long_index = pickup_long_index1.intersection(pickup_long_index2)

    pickup_lat_index1 = set(np.where(pickup_lat >= lat_min)[0])
    pickup_lat_index2 = set(np.where(pickup_lat <= lat_max)[0])
    pickup_lat_index = pickup_lat_index1.intersection(pickup_lat_index2)

    pickup_index = pickup_long_index.intersection(pickup_lat_index)
    return pickup_index


pickup_index = clean(pickup_long, pickup_lat)
dropoff_index = clean(dropoff_long, dropoff_lat)
final_index = np.array(list(pickup_index.intersection(dropoff_index)))
pickup_final = pickup[final_index, :]
dropoff_final = dropoff[final_index, :]
print(pickup_final.shape)
time_pickup = np.array(time_pickup)[final_index]
time_dropoff = np.array(time_dropoff)[final_index]

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

pickup_timestamp = turn_format(time_pickup)
dropoff_timestamp = turn_format(time_dropoff)
ptime_min = np.min(pickup_timestamp)
dtime_min = np.min(dropoff_timestamp)
ptime_max = np.max(pickup_timestamp)
dtime_max = np.max(dropoff_timestamp)
time_min = min(ptime_min, dtime_min)
time_max = max(ptime_max, dtime_max)
print(time_min)
print(time_max)
time_gap = 180*60
time_num = int((datetime.timestamp(time_max)-datetime.timestamp(time_min))//time_gap)  # 3h
print(time_num)

bounds = [long_min - 0.01, lat_min - 0.01, long_max + 0.01, lat_max + 0.01]
print("bounds:", bounds)
grid_length = 50
def geodistance(lng1, lat1, lng2, lat2):
    lng1, lat1, lng2, lat2 = map(radians, [float(lng1), float(lat1), float(lng2), float(lat2)])
    dlon = lng2 - lng1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    distance = 2 * asin(sqrt(a)) * 6371 * 1000
    distance = round(distance / 1000, 3)
    return distance


lat_length = geodistance(long_min, lat_min, long_min, lat_max) * 1000
lng_length = geodistance(long_min, lat_min, long_max, lat_min) * 1000
rows = int(lat_length / grid_length) + 1
cols = int(lng_length / grid_length) + 1

long_gap = float(abs(long_min - long_max) / cols)
lat_gap = float(abs(lat_min - lat_max) / rows)
poi_dict = collections.defaultdict(list)
pickup_dict = collections.defaultdict(list)
dropoff_dict = collections.defaultdict(list)
ptime_dict = collections.defaultdict(list)
dtime_dict = collections.defaultdict(list)

def vals(pickup_final, poi_dict, time_matrix=pickup_timestamp, time_dict=ptime_dict, type="pos"):
    pickup_index1 = np.array((pickup_final[:, 0] - long_min) / long_gap, dtype=int)
    pickup_index1 = np.clip(pickup_index1, 0, cols - 1)
    pickup_index2 = np.array((lat_max - pickup_final[:, 1]) / lat_gap, dtype=int)
    pickup_index2 = np.clip(pickup_index2, 0, rows - 1)
    for i in range(pickup_final.shape[0]):
        poi_dict[(pickup_index2[i], pickup_index1[i])].append(pickup_final[i, :])
        if type=="time":
            time_dict[(pickup_index2[i], pickup_index1[i])].append(time_matrix[i])


vals(poi_pos, poi_dict)
vals(pickup_final, pickup_dict, type="time")
vals(dropoff_final, dropoff_dict, time_matrix=dropoff_timestamp, time_dict=dtime_dict, type="time")

flow_dict = copy.deepcopy(pickup_dict)
for key in dropoff_dict:
    if key in flow_dict:
        flow_dict[key].extend(dropoff_dict[key])
    else:
        flow_dict[key] = dropoff_dict[key]

ftime_dict = copy.deepcopy(ptime_dict)
for key in dtime_dict:
    if key in ftime_dict:
        ftime_dict[key].extend(dtime_dict[key])
    else:
        ftime_dict[key] = dtime_dict[key]

# calculate median(average) flow and delete grids below it
print("num before:", len(poi_dict.keys()))
record_num = dict()
sums = []
for key in poi_dict:
    if key in flow_dict:
        record_num[key] = len(flow_dict[key])
        sums.append(len(flow_dict[key]))
    else:
        record_num[key] = 0
avg = np.mean(np.array(sums))
medium = np.median(np.array(sums))
print("average flow:", avg)
print("median flow:", medium)

for key in record_num:
    if record_num[key] < medium:
        del poi_dict[key]
print("num after:", len(poi_dict.keys()))


def plot_points(dicts, name):
    bounds = [long_min - 0.01, lat_min - 0.01, long_max + 0.01, lat_max + 0.01]
    fig = plt.figure(1, (8, 8), dpi=100)
    ax = plt.subplot(111)
    plt.sca(ax)
    fig.tight_layout(rect=(0.05, 0.1, 1, 0.9))
    # 背景
    plot_map.plot_map(plt, bounds, zoom=12, style=4)
    for key in dicts:
        plt.scatter(dicts[key][0], dicts[key][1], c="blue", s=1)
        # for item in dicts[key]:
        # plt.scatter(item[0],item[1],c = "blue" ,s= 1)
    plt.axis('off')
    plt.xlim(bounds[0], bounds[2])
    plt.ylim(bounds[1], bounds[3])
    plt.savefig(name)
    plt.show()


# plot_points(poi_dict)
dis = 75
directions = [[0, 1], [0, -1], [1, 0], [-1, 0], [-1, 1], [1, -1], [1, 1], [-1, -1]]
taxi_num_dict = dict()
taxi_sum_dict = dict()
for key in poi_dict:
    if len(poi_dict[key]) > 1:
        row, col = key
        sums = 0
        for poi in poi_dict[key]:
            nums = 0
            for i in range(8):
                row_new = row + directions[i][0]
                col_new = col + directions[i][1]
                if row_new >= 0 and row_new < rows and col_new >= 0 and col_new < cols:
                    key_new = (row_new, col_new)
                    if key_new in flow_dict:
                        taxis = flow_dict[key_new]
                        for taxi in taxis:
                            geo_dis = geodistance(poi[0], poi[1], taxi[0], taxi[1]) * 1000
                            if geo_dis <= dis:
                                nums += 1
            taxi_num_dict[(poi[0], poi[1])] = nums
            sums += nums
        taxi_sum_dict[key] = sums

poi_pos_final = dict()
for key in poi_dict:
    if len(poi_dict[key]) > 1:
        sums = taxi_sum_dict[key]
        long_new = 0
        lat_new = 0
        for item in poi_dict[key]:
            long_new = long_new + float(taxi_num_dict[(item[0], item[1])] / sums) * item[0]
            lat_new += float(taxi_num_dict[(item[0], item[1])] / sums) * item[1]
        poi_pos_final[key] = [long_new, lat_new]
    else:
        poi_pos_final[key] = poi_dict[key][0]
print("final len:", len(poi_pos_final.keys()))

time_features_dict = dict()
for key in poi_pos_final:
    row, col = key
    lng, lta = poi_pos_final[key]
    nums = 0
    if key not in time_features_dict:
        time_features_dict[key] = np.zeros(time_num+1)
    for i in range(8):
        row_new = row + directions[i][0]
        col_new = col + directions[i][1]
        if row_new >= 0 and row_new < rows and col_new >= 0 and col_new < cols:
            key_new = (row_new, col_new)
            if key_new in flow_dict:
                taxis = flow_dict[key_new]
                taxi_time = ftime_dict[key_new]
                for k in range(len(taxis)):
                    geo_dis = geodistance(lng, lta, taxis[k][0], taxis[k][1]) * 1000
                    if geo_dis <= dis:
                        index = int((datetime.timestamp(taxi_time[k]) - datetime.timestamp(time_min)) // time_gap)
                        time_features_dict[key][index]+=1

# times = 0
# for key in time_features_dict:
#     print(time_features_dict[key])
#     times+=1
#     if times>=5:
#         break

# plot_points(poi_pos_final, "points_final.png")
nodes_num = len(poi_pos_final.keys())
nodes = list(map(str, list(range(nodes_num))))
nonzero_dict = dict()
pos_dict = dict()
node_time = dict()
id = 0
G = nx.Graph()
for key in poi_pos_final:
    nonzero_dict[key] = nodes[id]
    pos_dict[nodes[id]] = poi_pos_final[key]
    node_time[nodes[id]] = time_features_dict[key]
    id+=1

edges = []
walk_dis = 300
for key in poi_pos_final:
    row, col = key
    row_left = max(0,row-10)
    row_right = min(row+10, rows-1)
    col_left = max(0,col-10)
    col_right = min(col+10, cols-1)
    for i in range(row_left, row_right+1):
        for j in range(col_left, col_right+1):
            if (i,j) in poi_pos_final:
                long1 = poi_pos_final[key][0]
                lat1 = poi_pos_final[key][1]
                long2 = poi_pos_final[(i,j)][0]
                lat2 = poi_pos_final[(i,j)][1]
                cal_dis = geodistance(long1, lat1, long2, lat2)*1000
                if cal_dis<=walk_dis:
                    edges.append((nonzero_dict[key], nonzero_dict[(i,j)]))

G.add_nodes_from(nodes)
G.add_edges_from(edges)
remove = []
for node in G.nodes():
    if G.degree(node)==0:
        remove.append(node)
G.remove_nodes_from(remove)
print("node number:", G.number_of_nodes())
print("edge number:", G.number_of_edges())

node_feature = {node: torch.tensor(list(node_time[node])) for node in G.nodes()}
nx.set_node_attributes(G, node_feature, name='node_feature')

# adjacency = nx.adjacency_matrix(G)
# pagerank = PageRank()
# scores = pagerank.fit_transform(adjacency)
# pos_features = np.zeros((len(poi_pos_final.keys()), 3))
# id = 0
# for nn in G.nodes:
#     pos_features[id,:2] = np.array(pos_dict[nn])
#     pos_features[id, 2] = scores[id]
#     id+=1

# np.savetxt("dblp.txt", pos_features)

# def build():
#     graph = Graph(G)
#     return graph
#
# def record_pos():
#     poses = []
#     for nn in G.nodes():
#         poses.append(pos_dict[nn])
#     return np.array(poses)
