# PTS

BFS: from line 133

DFS: from line 514

The bplanner code about beijing routes is in beijing_path: run 'python bplanner_v1.py'

line_station.json is what I generate, it is about the station information of each beijing line, but I find the number of station is over actual number for some routes. 

Here, I add the limitation of POI number to maximize it. Then consider choosing the topk of neighboring nodes would cost huge time, I mainly choose the all or most of the neighboring nodes of start point. Then for the second layer, choose the top-2 or top-3 neighboring nodes, after that, only choose the best point. However, all of these nodes are ensured to be reach destination.

