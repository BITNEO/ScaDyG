
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
feature_type = "one_hot"
#feature_type = "random"
''' Flights Transport 13,1691,927,145–&1 False 4months 122 days
 Can.Parl. Politics 734 74,478–&1 False 14years 14 years
 USLegis. Politics 225 60,396–&1 False 12congresses 12 congresses
 UNTrade Economics 255 507,497–&1 False 32years 32 years
 UNVote Politics 201 1,035,742–&1 False 72years 72 years
 Contact Proximity 692 2,426,279–&1 False 1month 8,064 5minutes'''
dataset = "UNvote"
path = "/home/wuxiang/DyGLib-master/DG_data/"+dataset+"/ml_"+dataset+".csv"
edge_path = "/home/wuxiang/DyGLib-master/DG_data/"+dataset+"/ml_"+dataset+ ".npy"
node_path = "/home/wuxiang/DyGLib-master/DG_data/"+dataset+"/ml_"+dataset+ "_node.npy"
if os.path.exists("./dataset/"+dataset)==False:
    os.system("mkdir /home/wuxiang/WinGNN-main/dataset/"+dataset)
    os.system("mkdir /home/wuxiang/WinGNN-main/dataset/"+dataset+"/edge_feature")
    os.system("mkdir /home/wuxiang/WinGNN-main/dataset/"+dataset+"/edge_index")
    os.system("mkdir /home/wuxiang/WinGNN-main/dataset/"+dataset+"/edge_time")
    os.system("mkdir /home/wuxiang/WinGNN-main/dataset/"+dataset+"/node_feature")
edge_feature = np.load(edge_path)[1:]
node_feautre = np.load(node_path)[1:]
f= open(path)
node_feautre = np.load(node_path)
lines = f.readlines()[1:]
lines = [line.split(',') for line in lines]
ts_list= [line[3] for line in lines]
max_ts = max(ts_list)

snapshot_ts_list = list(set(ts_list))
snapshot_ts_list = sorted([float(ts) for ts in snapshot_ts_list])

num_sanpshot = len(snapshot_ts_list)
#interval = int(float(max_ts)/num_sanpshot)

#timestamp_list = [i*interval for i in range(num_sanpshot)]

snapshot_list = []
edge_index = [[] for i in range(num_sanpshot)]
edge_feature_list = [[] for i in range(num_sanpshot)]
edge_time = [[] for i in range(num_sanpshot)]
print(len(lines))
scaler = StandardScaler()
for i,line in tqdm(enumerate(lines)):
    u = int(line[1])-1
    v = int(line[2])-1
    d_time = float(line[3])
    
    for k,time in enumerate(snapshot_ts_list):
        if d_time == snapshot_ts_list[k]:
            edge_index[k].append([u,v])
            

            #edge_feature_list[k].append(np.hstack((edge_feature[i],np.array(d_time))))
            edge_feature_list[k].append(edge_feature[i])
            edge_time[k].append(d_time)
    # 通过时间戳确定一个边应该分配到哪个快照（基于 snapshot_ts_intervals）
print(i)

for i in range(num_sanpshot):
    edge_feature_list[i] = scaler.fit_transform(edge_feature_list[i])

if feature_type == "one_hot":
    num_node = node_feautre.shape[0]
    indices = np.array([i for i in range(num_node)])

    # 使用np.eye生成一个二维数组，它的行数等于样本数，列数等于类别数
    # 然后使用indices选择对应的one-hot编码行
    one_hot_matrix = np.eye(num_node)[indices]
    one_hot_matrix = scaler.fit_transform(one_hot_matrix)
    node_feautre = one_hot_matrix

if feature_type == "random":
    node_feautre = np.load(node_path)

    random_vectors = np.random.randn(node_feautre.shape[0], 172)
    node_feautre = random_vectors

for i in range(num_sanpshot):
    np.save("/home/wuxiang/WinGNN-main/dataset/"+dataset+"/edge_index/"+str(i)+".npy",np.array(edge_index[i]).T)
    np.save("/home/wuxiang/WinGNN-main/dataset/"+dataset+"/edge_time/"+str(i)+".npy",np.array(edge_time[i]))
    np.save("/home/wuxiang/WinGNN-main/dataset/"+dataset+"/edge_feature/"+str(i)+".npy",np.array(edge_feature_list[i]))
    #np.save("/home/wuxiang/WinGNN-main/dataset/"+dataset+"/node_feature/"+str(i)+".npy",np.ones((node_feautre.shape[0],1)))
    np.save("/home/wuxiang/WinGNN-main/dataset/"+dataset+"/node_feature/"+str(i)+".npy",node_feautre)




            