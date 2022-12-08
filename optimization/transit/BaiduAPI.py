import requests
import json
import time
import numpy as np

def stod(o_Lat,o_Lng,d_Lng,d_Lat):#输入：起点纬度、起点经度、终点纬度、终点经度
#https://api.map.baidu.com/direction/v2/transit?origin=40.056878,116.30815&destination=31.222965,121.505821&tactics_incity	&ak=您的AK  //GET请求

    url ="http://api.map.baidu.com/direction/v2/transit?" #API地址
    ak = '7LGoRe9cnWfpUMZQPRPgfRtylQp7C1Km' #秘钥
    real_url = url +"origin="+o_Lat+","+o_Lng+"&destination="+d_Lat+","+d_Lng+"&tactics_incity="+1+"&ak="+ak #完整的请求代码
    '''
    tactics_incity默认为0
            可选值：
            0 推荐
            1 少换乘
            2 少步行
            3 不坐地铁
            4 时间短
            5 地铁优先
    '''
    req = requests.get(real_url)
    t = req.text
    data = json.loads(t) #将数据保存在数组data中
    try:#防止某几条数据报错导致请求终止
        total_duration = data['result']["routes"][0]["duration"]/60
        stepstr = str(total_duration) #获取该条数据总行程时间
        steps = data['result']["routes"][0]['steps'] #获取该条路径的所有步骤
        for step in steps:
            step_tructions = step[0]["instructions"]#每一步的介绍包括乘坐公交车或地铁线路名
            step_duration = step[0]["duration"]#每一步所花费的时间
            stepstr = stepstr+"/"+step_tructions+"/"+str(step_duration/60)
    except:
        stepstr = None
    req.close()
    return stepstr #返回数据为总时长/第一步/第一步耗时/第二步/第二步耗时/...



#test
#116.669438,39.903874
#116.682845,39.852909
url="https://api.map.baidu.com/direction/v2/transit?origin=39.903874,116.669438&destination=39.852909,116.682845&tactics_incity=1&ak=7LGoRe9cnWfpUMZQPRPgfRtylQp7C1Km"
req=requests.get(url)
t = req.text
data = json.loads(t) #将数据保存在数组data中

with open('title.json','w',encoding='utf8') as f2:
    # ensure_ascii=False才能输入中文，否则是Unicode字符
    # indent=2 JSON数据的缩进，美观
    json.dump(data,f2,ensure_ascii=False,indent=2)
data['result']["routes"][0]['steps']
#data['result']["routes"][0]['steps'].type
np.array(data['result']["routes"][0]['steps']).shape[0]
