'''
节点应包含
Lng,Lat
------------------------------
边应该包含
Transit=0,1(1表示此处发生了换乘,0则没有)
weight=实际距离
'''
import networkx as nx
import math
G = nx.Graph()
def Distance(Lat1, Lat2, Lng1, Lng2):
    radlat1 = rad(Lat1)
    radlat2 = rad(Lat2)
    radlng1 = rad(Lng1)
    radlng2 = rad(Lng2)
    a = radlat1 - radlat2
    b = radlng1 - radlng2
    #曼哈顿距离
    dis = 2 * 6378.137 * math.asin(math.sqrt(math.pow(math.sin(a/2), 2) + math.cos(radlat1) * math.cos(radlat2) * math.pow(math.sin(b/2), 2))) * 1000
    return dis

def add_transit(G,walk_distance_limit):
    for i,j in range(G.number_of_nodes()):
        dist=Distance(G[i]['Lat'],G[j]['Lat'],G[i]['Lng'],G[j]['Lng'])
        if(dist<=walk_distance_limit):
            G.add_edge(i,j,Transit=1,weight=dist)
        else:
            continue
def transit_routes(G,u,v):
    #u is source and v is target
    nx.shortest_path(G,u,v,weight=G.edges[u, v]['Dist'],method='dijkstra')