import numpy as np
import pandas as pd
import math
import gurobipy as gp
from gurobipy import GRB
import time
import folium
import folium.plugins as plugins


A = pd.read_csv('data/583.csv')
def rad(a):
    return a * math.pi / 180

def Distance(Lat1, Lat2, Lng1, Lng2):
    radlat1 = rad(Lat1)
    radlat2 = rad(Lat2)
    radlng1 = rad(Lng1)
    radlng2 = rad(Lng2)
    a = radlat1 - radlat2
    b = radlng1 - radlng2
    dis = 2 * 6378.137 * math.asin(math.sqrt(math.pow(math.sin(a/2), 2) + math.cos(radlat1) * math.cos(radlat2) * math.pow(math.sin(b/2), 2))) * 1000
    return dis

site = []
coordinate = {}
score = {}
for i in range(np.size(A,0)):
        sites = str(A.loc[i, 'PointName'])
        site.append(sites)
        coordinate[sites] = (float(A.loc[i, 'Lat']), float(A.loc[i, 'Lng']))
        score[sites] = float(A.loc[i, 'Score'])
        
        zoom = 16
        r = 5000
dist = {(c1, c2): Distance(coordinate[c1][0], coordinate[c2][0], coordinate[c1][1], coordinate[c2][1]) for c1 in site for c2 in site if c1 != c2}
for c1 in site:
    if c1 != '0':
        dis['0', c1] = 0
        dis[c1, '0'] = 0
    elif c1 != 'n+1':
        dis['n+1', c1] = 0
        dis[c1, 'n+1'] = 0

media_lat = 0
media_long = 0

for sites in site:
    media_lat = media_lat + coordinate[sites][0]
    media_long = media_long + coordinate[sites][1]

lat = media_lat / len(coordinate)
long = media_long / len(coordinate)


#Marker on the map: The nodes of the problem
import folium
map = folium.Map(location=[lat,long], zoom_start = zoom)
for sites in site:
    folium.Marker(location = coordinate[sites], tooltip = sites, icon=folium.Icon(color='darkred')).add_to(map)

# add search area circle
folium.Circle(radius=r, location=[lat,long], color='darkred').add_to(map)

map

#---------------MODELLAZIONE GUROBI--------------------------------
import numpy as np
import gurobipy as gp
from gurobipy import GRB

model = gp.Model('583_orienteering')

#Definition of decision variables决策变量
Xvars = model.addVars(dist.keys(), obj = score, vtype = GRB.BINARY, name = 'x') #xij
Yvars = model.addVars(site, obj = 0.0, vtype = GRB.BINARY, name = 'y')  #yi
#Definition of the objective function in GUROBI目标函数
obj = model.setObjective(gp.quicksum(Yvars[i]*score[i] for i in site), gp.GRB.MAXIMIZE)
start_point = site[0]
OutFirst = model.addConstr(Xvars.sum(start_point,'*') == 1)
end_point = site[len(site)-1]
InLast = model.addConstr(Xvars.sum('*',end_point) == 1)
# Budget constraint of the arcs on the intermediate nodes中间节点上的弧约束
Balance = model.addConstrs((gp.quicksum(Xvars.sum(i,j) for i in site if i != end_point)
                            == gp.quicksum(Xvars.sum(j,i) for i in site if i != start_point)
                            for j in site if i != j and j != start_point and j != end_point))
Visited = model.addConstrs((gp.quicksum(Xvars.sum(j,i) for i in site if i != start_point) == Yvars[j]
                            for j in site if i != j and j != end_point))
#DMAX线路长度约束
DMAX=10000
MaxDistance = model.addConstr((gp.quicksum(Xvars[i,j]*dist[i,j] for i in site for j in site if i != j) <= DMAX))
#FIXME-距离起点越来越远 距离终点越来越近
AwayStart = model.addConstr((dist[site[i]*Yvars[i],start_point]<dist[site[j]*Yvars[j],start_point]) for i in site for j in site if i != j))
CloseEnd = model.addConstr((dist[site[i]**Yvars[i],end_point]>dist[site[j]*Yvars[j],end_point]) for i in site for j in site if i != j))
#FIXME-Ziglag如果使用偏离起点的度数作为约束 现在使用15度
#AvoidZigzag = model.addConstr((gp.quicksum(int(dist[site[i], site[j]]>dist[site[i], site[i-1]]) for i in range(2, len(site)) for j in range(i-1))==(len(site)-1)*(len(site)-2)//2))
CosineAlg=model.addConstr((dist[start_point,site[i]*Yvars[i]]*dist[start_point,site[i]*Yvars[i]]+ dist[start_point,site[j]*Yvars[j]]*dist[start_point,site[j]*Yvars[j]] -dist[site[i]*Yvars[i],site[j]*Yvars[j]]*dist[site[i]*Yvars[i],site[j]*Yvars[j]]<0.9659258262890682867497431997289) for i,j in site if i != j))
#FIXME-不确定是for i,j in site if i != j  还是  for i in site for j in site if i != j

#MTZ约束
'''
If $x_{i,j} = 1$ then:
\begin{equation}
u_i + 1 = u_j \hspace{1cm} i,j = 1..n,i \neq j,j \neq 1
\end{equation}
'''
MTZ=1
if MTZ == 1:
    Uvars = model.addVars(site, vtype = GRB.CONTINUOUS, name = 'u')
    Postion = model.addConstrs((Xvars[i,j] == 1) >> (Uvars[i]+1==Uvars[j]) for i in site for j in site
                                 if j != start_point and i != j)

'''
#### Constraints of absence of 'Lazy Constraints'
As an alternative to the MTZ sub-turn constraints, ** Lazy Constraints ** can be added (modifying those offered by GUROBI appropriately)

\begin{equation}
    \sum_{(i,j) \epsilon S}^{} x_{ij} \le |S| - 1  \hspace{1cm} S \subset V \hspace{1cm} 2 \le |S| \le |V| - 1 
\end{equation}


The main **difference** with respect to the MTZ constraints is represented by the fact that the Lazy Constraints allow to solve the 'relaxed' model, that is, without the constraints of absence of sub-turns and to add them only when a current integer solution is found. This allows to consider a considerably lower number of constraints, which in the case of MTZ are polynomial number.
'''
#Functions to eliminate subtours (Callback)
def subtourelim(model, where):
    if where == GRB.Callback.MIPSOL:
        Xvals = model.cbGetSolution(model._Xvars)
        selected = gp.tuplelist((i,j) for i, j in model._Xvars.keys() if Xvals[i,j] > 0.5)
        tour = subtour(selected)

def subtour(edges):
    unvisited = list(site)
    while unvisited:
        thiscycle = []
        neighbors = unvisited
        while neighbors:
            current = neighbors[0]
            thiscycle.append(current)
            unvisited.remove(current)
            neighbors = [j for i, j in edges.select(current, '*')
                            if j in unvisited]
        if len(thiscycle) >= 2 and len(thiscycle) <= len(site)-1:
            model.cbLazy(gp.quicksum(model._Xvars[i,j] for i in thiscycle for j in thiscycle if i != j )
                         <= len(thiscycle)-1)

model.write('583_orienteering.lp')
model.write('583.mps')
import time

start = time.time()
MTZ = 1
if MTZ == 1:
    # model.Params.MIPGap = 0.10 #**50 *30 ***10
    model.optimize()
else:
    model._Xvars = Xvars
    model.Params.lazyConstraints = 1
    model.optimize(subtourelim)

end = time.time()
time_exec = round(end - start, 3)
print('Execution time ' + str(time_exec))
solution

#Function to build the path starting from the solution
def createtour(solution):
    unvisited = list(site)
    neighbors = unvisited
    tour = []
    while neighbors:
            current = neighbors[0]
            tour.append(current)
            unvisited.remove(current)
            neighbors = [j for i, j in solution.select(current, '*')
                            if j in unvisited]
    return tour


# Print of results
if model.status == GRB.OPTIMAL:

    foundOptimalSol = True
    Xvals = model.getAttr('x', Xvars)
    solution = gp.tuplelist((i, j) for i, j in Xvals.keys() if Xvals[i, j] > 0.5)
    Score = model.ObjVal

    # Print of the path obtained with the MTZs
    if MTZ == 1:
        Uvals = model.getAttr('x', Uvars)

        optTour = createtour(solution)
        path = list(optTour)

        print('Satisfaction value TOTAL - MTZ: %g' % Score)

    # Print of the path obtained with the Lazy Constraints
    else:
        optTour = createtour(solution)
        path = list(optTour)

        Score = 0
        for i in path:
            Score += score[i]

        print('Satisfaction value TOTAL - LAZY: %g' % Score)

    # Print of the number of Point of Interests
    print("N.POI: %g" % len(path))

    # Print of the total distance
    length = 0
    for i in solution:
        length += dist[i]
    print('Distance traveled in km : ' + str(length))
    print(path)


#Print tours
import folium
import folium.plugins as plugins

map = folium.Map(location=[lat, long], zoom_start=zoom)

points = []
for sites in path:
    points.append(coordinate[sites])
    if sites == start_point:  # start point
        folium.Marker(location=coordinate[sites], tooltip=sites,
                      icon=plugins.BeautifyIcon(icon="arrow-down", icon_shape="marker",
                                                border_color='#b22222',
                                                background_color='#b22222')).add_to(map)
    elif sites == end_point:  # end point
        folium.Marker(location=coordinate[sites], tooltip=sites,
                      icon=plugins.BeautifyIcon(icon="arrow-down", icon_shape="marker",
                                                border_color='#ffd700',
                                                background_color='#ffd700')).add_to(map)
    else:
        folium.Marker(location=coordinate[sites], tooltip=sites, icon=folium.Icon(color='darkred')).add_to(map)

folium.PolyLine(points, color='darkred').add_to(map)

map


conv_coord = []
mytour=[]
i=0
for sites in path:
    conv_coord.append((coordinate[sites][1],coordinate[sites][0]))
    mytour.append(list(conv_coord[i]))
    i+=1
#TODO 将路线映射到真实图像上去
# 1. gps坐标可能对照出现问题
# 2. 映射到真实路线上时会出现问题
# 3. 如果换用北京服务器图床的话暂时还不知道怎么操作
#Printing of the path for REAL ways
if (namefile == "beijing" ):
        import openrouteservice as ors
        import folium
        import folium.plugins as plugins

        # API Key di Open Route Service
        ors_key = '5b3ce3597851110001cf6248435cfcfbcf0c42858fde19dccf6f9c0f'
        client = ors.Client(key=ors_key)

        route = client.directions(coordinates=mytour,
                                  profile='foot-walking',
                                  format='geojson')


        map = folium.Map(location=[lat,long], zoom_start = zoom)
        #for sites in path:
            #folium.Marker(location = coordinate[sites], tooltip = sites).add_to(map)

        num = 0
        for sites in path:
            if sites == start_point: #start point
                folium.Marker(location = coordinate[sites], tooltip = sites,
                              icon=plugins.BeautifyIcon(icon="arrow-down", icon_shape="marker",
                                                        number=num,
                                                        border_color= '#b22222',
                                                        background_color='#b22222')).add_to(map)
            elif sites == end_point: #end point
                folium.Marker(location = coordinate[sites], tooltip = sites,
                              icon=plugins.BeautifyIcon(icon="arrow-down", icon_shape="marker",
                                                        number=num,
                                                        border_color= '#ffd700',
                                                        background_color='#ffd700')).add_to(map)
            else:
                folium.Marker(location = coordinate[sites], tooltip = sites,
                              icon=plugins.BeautifyIcon(icon="arrow-down", icon_shape="marker",
                                                        number=num,
                                                        border_color= '#b22222',
                                                        background_color='#ffffff')).add_to(map)
            num+=1

        # Aggiungo il GeoJson alla mappa
        folium.GeoJson(route, name=('Itinerario Scavi di Pompei con ' + str(DMAX) + ' ore'),
                       style_function=lambda feature: {'color': 'darkred'}).add_to(map)

        # Aggiungo il livello del path alla mappa
        folium.LayerControl().add_to(map)

        print('Distance traveled in km (real): ' + str((route['features'][0]['properties']['summary']['distance'])/1000))

map