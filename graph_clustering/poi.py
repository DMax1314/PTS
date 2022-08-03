import numpy as np
import csv
import os
import plot_map
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler

plot_map.set_mapboxtoken(r'pk.eyJ1IjoibHl1ZCIsImEiOiJjbDVhcjJ6Z3QwMGVwM2lxOGc1dGF0bmlmIn0.-7ibyu3eBAyD8EhBq_2h7g')
plot_map.set_imgsavepath(r'/Users/lyudonghang/Downloads/NYDATA/NYTaxi')

path="./POI"
long = []
lat = []
for file in os.listdir(path):
    if "POI" in file:
        path_new = os.path.join(path, file)
        times=0
        with open(path_new,'r') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                times+=1
                if times==1:
                    continue
                long.append(float(row[0]))
                lat.append(float(row[1]))

long = np.array(long[1:]).reshape(1,-1)
long_min = np.min(long)
long_max = np.max(long)
lat = np.array(lat[1:]).reshape(1,-1)
lat_min = np.min(lat)
lat_max = np.max(lat)
poi_pos = np.concatenate((long, lat), axis=0)
poi_pos = np.transpose(poi_pos)
print(poi_pos.shape)

poi_pos_new = StandardScaler().fit_transform(poi_pos)
db = DBSCAN(eps=0.1, min_samples=20).fit(poi_pos_new)
# db = KMeans(n_clusters=10, n_init=25).fit(poi_pos)
labels = db.labels_
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
print(n_clusters_)
print(labels[:10])
print(len(labels[labels==0]))

def plot_clusters_new(clusters, labels, cl):
    bounds = [long_min-0.01, lat_min-0.01, long_max+0.01, lat_max+0.01]
    fig     = plt.figure(1,(8,8),dpi = 100)    
    ax      = plt.subplot(111)
    plt.sca(ax)
    fig.tight_layout(rect = (0.05,0.1,1,0.9))
    #背景
    plot_map.plot_map(plt,bounds,zoom = 12,style = 4)
    print(len(labels[labels==cl]))
    for i in range(clusters.shape[0]):
        if labels[i]==cl:
            print(i)
            plt.scatter(clusters[i,0],clusters[i,1],c = "blue" ,s= 1)
    plt.axis('off')
    plt.xlim(bounds[0],bounds[2])
    plt.ylim(bounds[1],bounds[3])
    plt.savefig("./clusters/cluster_{}.png".format(cl))
    plt.show()

# for i in range(1):
#     plot_clusters_new(poi_pos, labels, i)

from sknetwork.data import karate_club
from sknetwork.ranking import PageRank

graph = karate_club(metadata=True)
adjacency = graph.adjacency
print(adjacency)
position = graph.position

pagerank = PageRank()
scores = pagerank.fit_transform(adjacency)
print(scores)