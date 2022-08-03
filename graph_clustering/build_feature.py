from datetime import datetime
from tqdm import tqdm
from math import radians, cos, sin, asin, sqrt
import seaborn as sns
import matplotlib
import numpy as np
from matplotlib.pyplot import *
from matplotlib import animation 
from matplotlib import cm
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from dateutil import parser
import io
import base64
import os
import pandas as pd

file_path = "cluster_data_250.csv"

Manhtun_taxi = pd.read_csv(file_path,parse_dates=['tpep_pickup_datetime','tpep_dropoff_datetime'])

taxi_data = Manhtun_taxi
taxi_data['timestamp'] = taxi_data['tpep_pickup_datetime'].apply(lambda x:datetime.timestamp(x))

def build_matrix(pickup, rows, cols):
    mt = np.zeros((rows, cols))
    assert pickup.size != 0,'errors'
    min_pickup_long = np.min(pickup[0,:])
    max_pickup_lat = np.max(pickup[1,:])
    max_pickup_long = np.max(pickup[0,:])
    min_pickup_lat = np.min(pickup[1,:])
    gap_long = float((max_pickup_long-min_pickup_long)/cols)
    gap_lat =  float((max_pickup_lat-min_pickup_lat)/rows)
    col_index = np.array((pickup[0,:]-min_pickup_long)/gap_long, dtype=int)
    row_index = np.array((max_pickup_lat-pickup[1,:])/gap_lat, dtype=int)
    for i in range(col_index.shape[0]):
        if row_index[i]==rows:
            row_index[i] = rows-1
        if col_index[i]==cols:
            col_index[i] = cols-1
        mt[row_index[i], col_index[i]] += 1
    return mt

def build_nodeCSv(one_matrix):
    return one_matrix.reshape(1,-1)

time_min = np.min(taxi_data['timestamp'])
date_time_min = datetime.fromtimestamp(time_min)
time_max = np.max(taxi_data['timestamp'])
date_time_max = datetime.fromtimestamp(time_max)

def build_time_feature(row, col):
    start = datetime(2015,1,1,0,0,0) #分割时间的开始
    startstamp = datetime.timestamp(start)
    gap_counter = int((date_time_max-date_time_min)//(15*60))
    for i in range(gap_counter):
        startstamp2 = startstamp + (i) * 15 * 60
        splitdata = taxi_data[taxi_data['timestamp'].between(startstamp2, startstamp2 + 15 * 60, inclusive=True)]
        pickup = np.array([splitdata.iloc[:, 4], splitdata.iloc[:, 5]], dtype=float)
        pickup_new = pickup.reshape(2, -1)
        pickup_matrix = build_matrix(pickup_new, row, col)
        taxi_split_data = pd.DataFrame(build_nodeCSv(pickup_matrix)).astype('int')