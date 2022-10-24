import geojson
import plot_map
import matplotlib.pyplot as plt
import folium

# plot_map.set_mapboxtoken(r'pk.eyJ1IjoibHl1ZCIsImEiOiJjbDVhcjJ6Z3QwMGVwM2lxOGc1dGF0bmlmIn0.-7ibyu3eBAyD8EhBq_2h7g')
# plot_map.set_imgsavepath(r'/Users/lyudonghang/PycharmProjects/beijing_path')

def plot_clusters_new(clusters_x, clusters_y, bounds):
    # bounds = [-75, 40, -73, 42]
    fig     = plt.figure(1,(8,8),dpi = 100)    
    ax      = plt.subplot(111)
    plt.sca(ax)
    fig.tight_layout(rect = (0.05,0.1,1,0.9))
    plot_map.plot_map(plt,bounds,zoom = 12,style = 4)
    plt.plot(clusters_x, clusters_y, color="red")
    plt.scatter(clusters_x[0], clusters_y[0],marker='*', color="blue")
    plt.scatter(clusters_x[-1], clusters_y[-1], marker='+', color="yellow")
    plt.axis('off')
    plt.xlim(bounds[0],bounds[2])
    plt.ylim(bounds[1],bounds[3])
    plt.show()

path = "./data/bus_line.geojson"
with open(path, 'r') as f:
    gj = geojson.load(f)
data = gj["features"]
# for i in range(1):
#     line = data[i]["geometry"]["coordinates"][0]
#     longs = []
#     lats = []
#     for j in range(len(line)):
#         longs.append(line[j][0])
#         lats.append(line[j][1])
#     longs_min = min(longs)
#     longs_max = max(longs)
#     lats_min = min(lats)
#     lats_max = max(lats)
#     bounds = [longs_min, lats_min, longs_max, lats_max]
#     # plot_clusters_new(longs, lats, bounds)
#     m = folium.Map([longs[0], lats[0]], zoom_start=10)
#     route = folium.PolyLine(line, weight=3, color="blue", opacity=0.8).add_to(m)
#     m.save("line.html")

for i in range(10):
    line = data[i]["geometry"]["coordinates"][0]
    print(len(line))
    if i==9:
        line_new = []
        for i in range(len(line)):
            line_new.append(line[i][::-1])
        m = folium.Map(line_new[0], zoom_start=14)
        route = folium.PolyLine(line_new, weight=3, color="blue", opacity=0.8).add_to(m)
        folium.Marker(line_new[0], popup='<b>Starting Point</b>').add_to(m)
        folium.Marker(line_new[-1], popup='<b>Ending Point</b>').add_to(m)
        m.save("line.html")