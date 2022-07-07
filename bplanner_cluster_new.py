import numpy as np
from copy import deepcopy
from collections import deque 
from math import radians, cos, sin, asin, sqrt
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import random
import datetime
import collections
import math
import networkx as nx
import plot_map 

plot_map.set_mapboxtoken(r'pk.eyJ1IjoibHl1ZCIsImEiOiJjbDVhcjJ6Z3QwMGVwM2lxOGc1dGF0bmlmIn0.-7ibyu3eBAyD8EhBq_2h7g')
plot_map.set_imgsavepath(r'/Users/lyudonghang/Downloads/NYDATA/NYTaxi')

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
passenger = np.array(datas_new[:, 3], dtype=int)
trip_distance = np.array(datas_new[:, 4], dtype=float)
t_pick = np.array(datas_new[:,1])
t_drop = np.array(datas_new[:,2])

def clean(pickup_long, pickup_lat):
    pickup_long_index1 = set(np.where(pickup_long>-75)[0])
    pickup_long_index2 = set(np.where(pickup_long<-73)[0])
    pickup_long_index = pickup_long_index1.intersection(pickup_long_index2)
    
    pickup_lat_index1 = set(np.where(pickup_lat>40)[0])
    pickup_lat_index2 = set(np.where(pickup_lat<42)[0])
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
print(dropout_final.shape)
passenger = passenger[final_index]
trip_distance = trip_distance[final_index]
t_pick = t_pick[final_index]
t_drop = t_drop[final_index]

def turn_format(t_pick):
    tms = []
    for i in range(t_pick.shape[0]):
        temp = t_pick[i].strip().split()
        temp1 = list(map(int, temp[0].split('-')))
        temp2 = list(map(int, temp[1].split(':')))
        temp1.extend(temp2)
        timec = datetime.datetime(temp1[0], temp1[1], temp1[2], temp1[3], temp1[4], temp1[5])
        tms.append(timec)
    return np.array(tms)

t_pick_time = turn_format(t_pick)
t_drop_time = turn_format(t_drop)
t_time = t_drop_time-t_pick_time
t_final_time = np.array(list(map(lambda x: x.seconds/60, t_time)))
print("time:", t_final_time[:10])

min_long = min(np.min(pickup_final[0,:]), np.min(dropout_final[0,:]))
max_lat = max(np.max(pickup_final[1,:]), np.max(dropout_final[1,:]))
max_long = max(np.max(pickup_final[0,:]), np.max(dropout_final[0,:]))
min_lat = min(np.min(pickup_final[1,:]), np.min(dropout_final[1,:]))
print(min_lat, min_long, max_lat, max_long)


def geodistance(lng1,lat1,lng2,lat2):
    lng1, lat1, lng2, lat2 = map(radians, [float(lng1), float(lat1), float(lng2), float(lat2)])
    dlon=lng2-lng1
    dlat=lat2-lat1
    a=sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    distance=2*asin(sqrt(a))*6371*1000
    distance=round(distance/1000,3)
    return distance

lat_length = geodistance(min_long, min_lat, min_long, max_lat)
lng_length = geodistance(min_long, min_lat, max_long, min_lat)
# 100m x 100m
rows = int(lat_length*10)+1
cols = int(lng_length*10)+1
pdr_matrix = np.zeros((rows, cols))
pdr_time_matrix = np.zeros((rows, cols))
print(pdr_matrix.shape)

long_gap = float(abs(min_long-max_long)/cols)
lat_gap = float(abs(min_lat-max_lat)/rows)

def vals(pickup_final):
    pickup_index1 = np.array((pickup_final[0,:]-min_long)/long_gap, dtype=int)
    pickup_index1 = np.clip(pickup_index1, 0, cols-1)
    pickup_index2 = np.array((max_lat-pickup_final[1,:])/lat_gap, dtype=int)
    pickup_index2 = np.clip(pickup_index2, 0, rows-1)
    for i in range(pickup_final.shape[1]):
        pdr_matrix[pickup_index2[i], pickup_index1[i]]+=1
        pdr_time_matrix[pickup_index2[i], pickup_index1[i]]+=t_final_time[i]

vals(pickup_final)
vals(dropout_final)

# in the original paper, PDRs per hour > 0.2, it would cause too many isolated hot points and make number of candidated bus stop large
# here choose PDRs>15

# pdr_time_matrix[pdr_time_matrix==0]=1
# pdr_matrix = pdr_matrix/(pdr_time_matrix/60)
hot_threshold = 15
print(pdr_matrix[pdr_matrix>hot_threshold].shape)
print("ratio:", float(pdr_matrix[pdr_matrix>hot_threshold].shape[0]/(rows*cols)))

# use BFS algorithm to find hot partitions
partition_matrix = np.zeros((rows, cols))
judge_matrix = np.zeros((rows, cols))
partition_matrix[pdr_matrix>hot_threshold] = 1
directions = [[0,1],[0,-1],[1,0],[-1,0],[-1,1],[1,-1],[1,1],[-1,-1]]

def BFS(partition_matrix, judge_matrix, q):
    subs = []
    long_subs = []
    lat_subs = []
    while len(q)>0:
        row, col = q.popleft()
        subs.append(pdr_matrix[row, col])
        long_subs.append(min_long+long_gap*(col+0.5))
        lat_subs.append(max_lat-lat_gap*(row+0.5))
        for dd in directions:
            row_new = row+dd[0]
            col_new = col+dd[1]
            if row_new>=0 and row_new<rows and col_new>=0 and col_new<cols and partition_matrix[row_new, col_new]==1 and judge_matrix[row_new, col_new]==0:
                q.append([row_new, col_new])
                judge_matrix[row_new, col_new] = 1
    return judge_matrix, subs, long_subs, lat_subs
             

partitions = []
partitions_long = []
partitions_lat = []
for i in range(partition_matrix.shape[0]):
    for j in range(partition_matrix.shape[1]):
        if partition_matrix[i,j]==1 and judge_matrix[i,j]==0:
            q = deque()
            q.append([i, j])
            judge_matrix, subs, long_subs, lat_subs = BFS(partition_matrix, judge_matrix, q)
            partitions.append(subs)
            partitions_long.append(long_subs)
            partitions_lat.append(lat_subs)
print(len(partitions))

locs = []
def calculate_loc(partitions, partitions_long, partitions_lat):
    wsums_long = 0
    wsums_lat = 0
    sums = sum(partitions)
    for j in range(len(partitions)):
        wsums_long = wsums_long+partitions[j]*partitions_long[j]
        wsums_lat = wsums_lat+partitions[j]*partitions_lat[j]
    loc_ave_long = float(wsums_long/sums)
    loc_ave_lat = float(wsums_lat/sums)
    return loc_ave_long, loc_ave_lat

for i in range(len(partitions)):
    loc_ave_long, loc_ave_lat = calculate_loc(partitions[i], partitions_long[i], partitions_lat[i])
    locs.append([loc_ave_long, loc_ave_lat])

length = []
for i in range(len(partitions)):
    length.append(len(partitions[i]))
indexes = np.argsort(np.array(length))[::-1]
print("maximal length:",len(partitions[indexes[1]]))

################## Merge Algorithm ########################
indexes_copy = deepcopy(indexes)
clusters = []
clusters_long = []
clusters_lat = []
T1 = 300
ii = len(partitions)
while ii > 0:
    cluster_tmp = deepcopy(partitions[indexes_copy[0]])
    cluster_long_tmp = deepcopy(partitions_long[indexes_copy[0]])
    cluster_lat_tmp = deepcopy(partitions_lat[indexes_copy[0]])
    ii -= 1
    old_long, old_lat = locs[indexes_copy[0]]
    indexes_copy = np.delete(indexes_copy, 0)
    jj = 0
    while jj < indexes_copy.shape[0]:
        cur_long, cur_lat = locs[indexes_copy[jj]]
        dist = geodistance(old_long, old_lat, cur_long, cur_lat)*1000
        if dist<T1:
            cluster_tmp.extend(partitions[indexes_copy[jj]])
            cluster_long_tmp.extend(partitions_long[indexes_copy[jj]])
            cluster_lat_tmp.extend(partitions_lat[indexes_copy[jj]])
            old_long, old_lat = calculate_loc(cluster_tmp, cluster_long_tmp, cluster_lat_tmp)
            indexes_copy = np.delete(indexes_copy,jj)
            ii -= 1
        else:
            jj+=1
    clusters.append(cluster_tmp)
    clusters_long.append(cluster_long_tmp)
    clusters_lat.append(cluster_lat_tmp)
print("length of cluster:", len(clusters))

def plot_clusters(clusters, clusters_long, clusters_lat):
    pre_matrix = np.zeros((rows, cols))
    val = 400
    for i in range(len(clusters)):
        cindex1 = np.array((clusters_long[i]-min_long)/long_gap-0.5, dtype=int)
        cindex1 = np.clip(cindex1, 0, cols-1)
        cindex2 = np.array((max_lat-clusters_lat[i])/lat_gap-0.5, dtype=int)
        cindex2 = np.clip(cindex2, 0, rows-1)
        for k in range(cindex1.shape[0]):
            pre_matrix[cindex2[k], cindex1[k]] = val
        val -= 1
    # sns.heatmap(pre_matrix)
    # # plt.show()
    # sns.heatmap(partition_matrix)
    # plt.show()

def plot_clusters_new(clusters_long, clusters_lat):
    bounds = [-75, 40, -73, 42]
    fig     = plt.figure(1,(8,8),dpi = 100)    
    ax      = plt.subplot(111)
    plt.sca(ax)
    fig.tight_layout(rect = (0.05,0.1,1,0.9))
    #背景
    plot_map.plot_map(plt,bounds,zoom = 12,style = 4)
    #colorbar
    # pallete_name = "autumn_r"
    # colors = sns.color_palette(pallete_name, 3)
    # print(colors)
    # print(len(colors))
    # colors.reverse()
    # cmap = mpl.colors.LinearSegmentedColormap.from_list(pallete_name, colors)
    cc = list(range(1,len(clusters_long)+1))
    random.shuffle(cc)
    # norm = mpl.colors.Normalize(vmin=0, vmax=vmax)
    #plot scatters
    for i in range(len(clusters_long)):
        plt.scatter(clusters_long[i],clusters_lat[i],c = "blue" ,s= 1)
    plt.axis('off')
    plt.xlim(bounds[0],bounds[2])
    plt.ylim(bounds[1],bounds[3])
    plt.show()


# plot_clusters(clusters, clusters_long, clusters_lat)
plot_clusters_new(clusters_long, clusters_lat)

####################### Split Algorithm ######################
def split_horizon(group_matrix):
    sums = np.sum(group_matrix)
    col_sum = np.sum(group_matrix, axis=1)
    balance_point = float(sums/2)
    sum_tmp = 0
    judge_tmp = float('inf')
    for i in range(col_sum.shape[0]):
        sum_tmp+=col_sum[i]
        if abs(balance_point-sum_tmp)<judge_tmp:
            judge_tmp = abs(balance_point-sum_tmp)
        else:
            return i

def split_vertical(group_matrix):
    sums = np.sum(group_matrix)
    row_sum = np.sum(group_matrix, axis=0)
    balance_point = float(sums/2)
    sum_tmp = 0
    judge_tmp = float('inf')
    for i in range(row_sum.shape[0]):
        sum_tmp+=row_sum[i]
        if abs(balance_point-sum_tmp)<judge_tmp:
            judge_tmp = abs(balance_point-sum_tmp)
        else:
            return i


# split, record the longtitude and latitude of left upper corner and right bottom corner such that estimate the geographical coordinate
# of hot points
def split_cluster(group_matrix, clusters_final, fourpoints, left, up, lng_lat):
    if group_matrix.shape[0]<=10 and group_matrix.shape[1]<=10:
        long_min, lat_max, ln_gap, lat_gap = lng_lat
        # avoid all zeros
        if group_matrix.any():
            clusters_final.append(group_matrix)
            fourpoints.append([float(long_min+ln_gap*left), float(lat_max-lat_gap*up), ln_gap, lat_gap])
        return
    elif group_matrix.shape[1]<=10:
        row_indexs = split_horizon(group_matrix)
        split_cluster(group_matrix[:row_indexs, :], clusters_final, fourpoints, left, up, lng_lat)
        split_cluster(group_matrix[row_indexs:, :], clusters_final, fourpoints, left, up+row_indexs, lng_lat)
    elif group_matrix.shape[0]<=10:
        col_index = split_vertical(group_matrix)
        split_cluster(group_matrix[:, :col_index], clusters_final, fourpoints, left, up, lng_lat)
        split_cluster(group_matrix[:, col_index:], clusters_final, fourpoints, left+col_index, up, lng_lat)
    else:
        row_indexs = split_horizon(group_matrix)
        group_matrix1 = group_matrix[:row_indexs]
        group_matrix2 = group_matrix[row_indexs:]
        col_index1 = split_vertical(group_matrix1)
        col_index2 = split_vertical(group_matrix2)
        split_cluster(group_matrix1[:, :col_index1], clusters_final, fourpoints, left, up, lng_lat)
        split_cluster(group_matrix1[:, col_index1:], clusters_final, fourpoints, left+col_index1, up, lng_lat)
        split_cluster(group_matrix2[:, :col_index2], clusters_final, fourpoints, left, up+row_indexs, lng_lat)
        split_cluster(group_matrix2[:, col_index2:], clusters_final, fourpoints, left+col_index2, up+row_indexs, lng_lat)

def corner_points(longs, lats):         
    long_max = np.max(np.array(longs))
    long_min = np.min(np.array(longs))
    lat_max = np.max(np.array(lats))
    lat_min = np.min(np.array(lats))
    return long_max, long_min, lat_max, lat_min
    
clusters_final = []
fourpoints = []
T2=1000
for i in range(len(clusters)):
    temp = clusters[i]
    long_max, long_min, lat_max, lat_min = corner_points(clusters_long[i], clusters_lat[i])
    width = geodistance(long_min, lat_max, long_max, lat_max)*1000
    height = geodistance(long_min, lat_max, long_min, lat_min)*1000
    rows_new = int(height/100)+1
    cols_new = int(width/100)+1
    ln_gap1 = float((long_max-long_min)/cols_new)
    lat_gap1 = float((lat_max-lat_min)/rows_new)
    group_matrix = np.zeros((rows_new, cols_new))
    index1 = np.array((np.array(clusters_long[i])-long_min)/ln_gap1-0.5, dtype=int)
    index1 = np.clip(index1, 0, cols_new-1)
    index2 = np.array((lat_max-np.array(clusters_lat[i]))/lat_gap1-0.5, dtype=int)
    index2 = np.clip(index2, 0, rows_new-1)
    for j in range(len(temp)):
        group_matrix[index2[j], index1[j]] = temp[j]
    
    if width<=T2 and height<=T2:
        clusters_final.append(group_matrix)
        fourpoints.append([long_min, lat_max, ln_gap1, lat_gap1])
    else:
        split_cluster(group_matrix, clusters_final, fourpoints, 0, 0, [long_min, lat_max, ln_gap1, lat_gap1])

# choose candidated bus stops for each hot partitions (isolated hot point is candidated result directly)
def select_candidate_stop(grid, fourpoints):
    scores = 0
    long_min, lat_max, ln_gap, lat_gap = fourpoints
    long_c = long_min
    lat_c = lat_max
    if ln_gap!=0 and lat_gap!=0:
        pdrs = np.sum(grid)
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if grid[i,j]!=0:
                    cd = 0
                    for tt in directions:
                        i_new = i+tt[0]
                        j_new = j+tt[1]
                        if i_new>=0 and i_new<grid.shape[0] and j_new>=0 and j_new<grid.shape[1] and grid[i_new, j_new]!=0:
                            cd+=1
                    score = 0.5*float((cd+1)/9)+0.5*float(grid[i,j]/pdrs)
                    if score>scores:
                        long_c = float(long_min+ln_gap*(j+0.5))
                        lat_c = float(lat_max-lat_gap*(i+0.5))
    return long_c,lat_c
    
candidate_bustop = []
for i in range(len(clusters_final)):
    grids = clusters_final[i]
    long_c, lat_c = select_candidate_stop(grids, fourpoints[i])
    candidate_bustop.append([long_c, lat_c])
print(candidate_bustop[:10])
print("length of candidate bus stops:", len(candidate_bustop))

############################# Second Phase ################################
def return_pos_new(long, lat):
    col1 = int((long-min_long)/long_gap-0.5)
    col1 = min(col1, cols-1)
    row1 = int((max_lat-lat)/lat_gap-0.5)
    row1 = min(row1, rows-1)
    return col1, row1

def return_pos(long, lat):
    col1 = int((long-min_long)/long_gap)
    col1 = min(col1, cols-1)
    row1 = int((max_lat-lat)/lat_gap)
    row1 = min(row1, rows-1)
    return col1, row1

poses = []
for i in range(len(candidate_bustop)):
    col1, row1 = return_pos_new(candidate_bustop[i][0], candidate_bustop[i][1])
    poses.append([col1, row1])

start_row = []
end_col = []
records_tm = collections.defaultdict(list)
records_fm = collections.defaultdict(list)
records_trip = collections.defaultdict(list)
for i in range(pickup_final.shape[1]):
    col2, row2 = return_pos(pickup_final[0, i], pickup_final[1, i])
    col3, row3 = return_pos(dropout_final[0, i], dropout_final[1, i])
    key = (col2, row2, col3, row3)  # 元组
    if [col2, row2] in poses and [col3, row3] in poses:
        records_tm[key].append(t_final_time[i])
        records_fm[key].append(passenger[i])
        records_trip[key].append(trip_distance[i])
        if [col2, row2] not in start_row:
            start_row.append([col2, row2])
        if [col3, row3] not in end_col:
            end_col.append([col3, row3])

# build od matrix, tm matrix and fm matrix based on getting candidated bus stops, also build graph by networkx
rr = len(start_row)
cc = len(end_col)
od_matrix = np.zeros((rr, cc))
print("OD matrix:", od_matrix.shape)
tm_matrix = np.zeros((rr, cc))
fm_matrix = np.zeros((rr, cc))
nodes_tmp = list(range(len(candidate_bustop)))
nodes = list(map(str, nodes_tmp))
edges = []
for key in records_tm:
    row_tmp = start_row.index(list(key[:2]))
    node_start = poses.index(list(key[:2]))
    col_tmp = end_col.index(list(key[2:]))
    node_end = poses.index(list(key[2:]))
    tm_matrix[row_tmp, col_tmp] = sum(records_tm[key])/len(records_tm[key])
    fm_matrix[row_tmp, col_tmp] = sum(records_fm[key])/len(records_fm[key])
    od_matrix[row_tmp, col_tmp] = len(records_tm[key])
    if node_start!=node_end:
        edges.append((str(node_start), str(node_end)))

G = nx.DiGraph()
G.add_nodes_from(nodes)
G.add_edges_from(edges)

def cal_long(col):
    geo_lng = min_long+long_gap*(col+random.uniform(0.4,0.6))
    return geo_lng

def cal_col(lng):
    return (lng-min_long-random.uniform(0.4, 0.6)*long_gap)/long_gap

def cal_lat(row):
    geo_lat = max_lat-lat_gap*(row+random.uniform(0.4,0.6))
    return geo_lat

def cal_row(lat):
    return -((lat-max_lat+random.uniform(0.4, 0.6)*lat_gap)/lat_gap)

def trans(index):
    col, row = poses[int(index)]
    return cal_long(col), cal_lat(row)

# implement DFS to search routes between origin and destination based on five criterions
def judge_line(node_start_tmp, node_end_tmp):
    geo_lng, geo_lat = trans(node_start_tmp)
    geo_lng1, geo_lat1 = trans(node_end_tmp)

    theta = math.atan((geo_lat1-geo_lat)/(geo_lng1-geo_lng))
    epsi = 1.5
    length_threshold = 20
    def criter1(line):
        line = list(map(lambda x:trans(x), line))
        for i in range(1, len(line)):
            # print("distance:", geodistance(line[i][0], line[i][1], line[i-1][0], line[i-1][1]))
            if geodistance(line[i][0], line[i][1], line[i-1][0], line[i-1][1])>epsi:
                return False
        return True
        
    def criter2(line):
        line = list(map(lambda x:trans(x), line))
        line = np.array(line)
        line_new = line[1:line.shape[0], 0]*math.cos(theta)+line[1:line.shape[0], 1]*math.sin(theta)
        # print("line_new:", line_new)
        return all([line_new[i] < line_new[i+1] for i in range(line_new.shape[0]-1)])

    def criter3(line):
        line = list(map(lambda x:trans(x), line))
        dis_diff = list(map(lambda x: geodistance(geo_lng, geo_lat, x[0], x[1]), line[1:len(line)]))
        # print("dis_diff_start:", dis_diff)
        return all([dis_diff[i] < dis_diff[i+1] for i in range(len(dis_diff)-1)])

    def criter4(line):
        line = list(map(lambda x:trans(x), line))
        dis_diff = list(map(lambda x: geodistance(geo_lng1, geo_lat1, x[0], x[1]), line[1:len(line)]))
        # print("dis_diff_end:", dis_diff)
        return all([dis_diff[i] > dis_diff[i+1] for i in range(len(dis_diff)-1)])

    def criter5(line):
        line = list(map(lambda x:trans(x), line))
        for i in range(2, len(line)):
            lng_tmp = line[i][0]
            lat_tmp = line[i][1]
            dis_diff = np.array(list(map(lambda x: geodistance(lng_tmp, lat_tmp, x[0], x[1]), line[:i])))
            if np.argmin(dis_diff)!=dis_diff.shape[0]-1:
                return False
        return True
 
    # DFS
    lines = []
    def explore(starts, ends, line):
        if len(lines)>15:
            return
        if starts==ends:
            lines.append(deepcopy(line))
            print("suitable line:",deepcopy(line))
            return
        if len(line)>length_threshold:
            return
        for node in G.neighbors(starts):
            if len(line)>=3 and criter1(line) and criter2(line) and criter3(line) and criter4(line) and criter5(line):
                line.append(node)
                explore(node, ends, line)
                line.pop()
            elif len(line)<3:
                line.append(node)
                explore(node, ends, line)
                line.pop()
    node_start_tmp = str(node_start_tmp)
    node_end_tmp = str(node_end_tmp)
    explore(node_start_tmp, node_end_tmp, [node_start_tmp])
    return lines

# find suitable pair of origin and destination
'''
judge = True
starts = 0
ends = 0
oppo_starts = 0
oppo_ends = 0
epsi = 1.8
while judge:
    starts = random.randint(0,rr-1)
    ends = random.randint(0,cc-1)
    try:
        oppo_starts = start_row.index(end_col[ends])
        oppo_ends = end_col.index(start_row[starts])
    except:
        continue
    if od_matrix[starts, ends]==0:
        col, row = start_row[starts]
        node_start_tmp = str(poses.index([col, row]))
        geo_lng = min_long+long_gap*(col+random.uniform(0.45,0.55))
        geo_lat = max_lat-lat_gap*(row+random.uniform(0.45,0.55))
        colt, rowt = end_col[ends]
        node_end_tmp = str(poses.index([colt, rowt]))
        geo_lng1 = min_long+long_gap*(colt+random.uniform(0.45,0.55))
        geo_lat1 = max_lat-lat_gap*(rowt+random.uniform(0.45,0.55))
        try:
            num = nx.shortest_path_length(G, source=node_start_tmp, target=node_end_tmp)
            if geodistance(geo_lng, geo_lat, geo_lng1, geo_lat1)>10 and geodistance(geo_lng, geo_lat, geo_lng1, geo_lat1)<20 and num>=4:
                lines = judge_line(node_start_tmp, node_end_tmp)
                if len(lines)>0:
                    judge = False
        except:
            pass


print("starts:", node_start_tmp)
print("ends:", node_end_tmp)
'''

# some pairs of origin and destination have no suitable routes for fitting all the criterions, while some pairs
# would have with too many routes and take too long time with DFS even if using all the criterions to speed up convergence
node_start_tmp = str(171)
node_end_tmp = str(45)
lines = judge_line(node_start_tmp, node_end_tmp)
print(len(lines))
print(lines)

demo_line = lines[10]
xx = []
yy = []
for i in range(len(demo_line)): 
    demo_col, demo_row = poses[int(demo_line[i])]
    xx.append(cal_long(demo_col))
    yy.append(cal_lat(demo_row))
plt.plot(xx, yy)
plt.show()

