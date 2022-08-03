import numpy as np
import csv
import plot_map
import matplotlib.pyplot as plt
import folium
from sklearn.cluster import DBSCAN, KMeans, OPTICS

plot_map.set_mapboxtoken(r'pk.eyJ1IjoibHl1ZCIsImEiOiJjbDVhcjJ6Z3QwMGVwM2lxOGc1dGF0bmlmIn0.-7ibyu3eBAyD8EhBq_2h7g')
plot_map.set_imgsavepath(r'/Users/lyudonghang/Downloads/NYDATA/NYTaxi')
path_new = "Bus_Stop_Shelter.csv"
bus_lng = []
bus_lat = []
times = 0
with open(path_new, 'r') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        times+=1
        if times==1:
            continue
        bus_lng.append(row[-5])
        bus_lat.append(row[-4])

bus_lng = np.array(bus_lng,dtype=float)
bus_lat = np.array(bus_lat, dtype=float)
print(bus_lat.shape)

bounds = [-74.0282068, 40.6957492, -73.9196656, 40.8368934]
def clean(pickup_long, pickup_lat):
    pickup_long_index1 = set(np.where(pickup_long >= bounds[0])[0])
    pickup_long_index2 = set(np.where(pickup_long <= bounds[2])[0])
    pickup_long_index = pickup_long_index1.intersection(pickup_long_index2)

    pickup_lat_index1 = set(np.where(pickup_lat >= bounds[1])[0])
    pickup_lat_index2 = set(np.where(pickup_lat <= bounds[3])[0])
    pickup_lat_index = pickup_lat_index1.intersection(pickup_lat_index2)

    pickup_index = pickup_long_index.intersection(pickup_lat_index)
    return pickup_index

final_index = clean(bus_lng, bus_lat)
bus_lng = bus_lng[np.array(list(final_index))].reshape(1,-1)
bus_lat = bus_lat[np.array(list(final_index))].reshape(1,-1)
coor = np.concatenate((bus_lng, bus_lat), axis=0)
coor = np.transpose(coor)

db = OPTICS(min_samples=20).fit(coor)
labels = db.labels_
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
print(n_clusters_)
coor_0 = coor[labels==4]
def plot_points(arrs, name):
    fig = plt.figure(1, (8, 8), dpi=100)
    ax = plt.subplot(111)
    plt.sca(ax)
    fig.tight_layout(rect=(0.05, 0.1, 1, 0.9))
    # 背景
    plot_map.plot_map(plt, bounds, zoom=12, style=4)
    plt.scatter(arrs[:,0], arrs[:,1], c="red", s=1.5, marker='*')
    plt.axis('off')
    plt.xlim(bounds[0], bounds[2])
    plt.ylim(bounds[1], bounds[3])
    plt.show()

plot_points(coor_0, "bus_stop.png")
# city_map = folium.Map(location=[bounds[1], bounds[0]], zoom_start=10)
# city_map.save("test.html")