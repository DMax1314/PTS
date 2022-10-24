import geojson
import json

station_path = "./data/bus_station.geojson"
line_path = "./data/bus_line.geojson"

with open(station_path,'r') as f:
    gj = geojson.load(f)
data = gj["features"]
stations = []
for i in range(len(data)):
    pos = data[i]["geometry"]["coordinates"]
    stations.append(pos)

with open(line_path, 'r') as f:
    gj = geojson.load(f)
data = gj["features"]
lines = dict()
for i in range(len(data)):
    key = data[i]["properties"]["NAME"]
    val = data[i]["geometry"]["coordinates"][0]
    staion_new = []
    for point in val:
        if point in stations:
            staion_new.append(point)
    lines[key]=staion_new

outfile = "./data/line_station.json"
with open(outfile,'w', encoding="utf-8") as f:
    json.dump(lines,f,ensure_ascii=False,indent = 4)
