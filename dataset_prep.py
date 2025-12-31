#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Description :load datasets

import os
import copy
import math
import torch
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from tgb.utils.pre_process import load_label_dict

from tgb.utils.utils import  load_pkl
def load(nodes_num):
    """
    load_dataset
    :param nodes_num:
    :return:
    """
    path = "dataset/dblp_timestamp/"

    train_e_feat_path = path + 'train_e_feat/' + type + '/'
    test_e_feat_path = path + 'test_e_feat/' + type + '/'

    train_n_feat_path = path + type + '/' + 'train_n_feat/'
    test_n_feat_path = path + type + '/' + 'test_n_feat/'


    path = path + type
    train_path = path + '/train/'
    test_path = path + '/test/'

    train_n_feat = read_e_feat(train_n_feat_path)
    test_n_feat = read_e_feat(test_n_feat_path)

    train_e_feat = read_e_feat(train_e_feat_path)
    test_e_feat = read_e_feat(test_e_feat_path)

    num = 0
    train_graph = read_graph(train_path, nodes_num, num)
    num = num + len(train_graph)
    test_graph = read_graph(test_path, nodes_num, num)
    return train_graph, train_e_feat, train_n_feat, test_graph, test_e_feat, test_n_feat


def load_r(name):
    path = "dataset/" + name 
    path_ei = path + '/' + 'edge_index/'
    path_nf = path + '/' + 'node_feature/'
    path_ef = path + '/' + 'edge_feature/'
    path_et = path + '/' + 'edge_time/'

    edge_index = read_npz(path_ei)
    edge_feature = read_npz(path_ef)
    node_feature = read_npz(path_nf)
    edge_time = read_npz(path_et)
    all_edge_time = np.hstack(edge_time)
    unique_elements, counts = np.unique(all_edge_time, return_counts=True)
    tims_span =(max(unique_elements) - min(unique_elements))/(3600*24)
    #print("unique time:{}".format(counts))


    nodes_num = node_feature[0].shape[0]

    sub_graph = []
    for e_i in edge_index:
        row = e_i[0]
        col = e_i[1]
        ts = [1] * len(row)
        sub_g = coo_matrix((ts, (row, col)), shape=(nodes_num, nodes_num))
        sub_graph.append(sub_g)

    return sub_graph, edge_feature, edge_time, node_feature

def load_r_without_node(name):
    path = "./dataset/node_property_pred/" + name
    path_ei = path + '/' + 'edge_index/'
    path_nf = path + '/' + 'node_feature/'
    path_ef = path + '/' + 'edge_feature/'
    path_et = path + '/' + 'edge_time/'
    path_nl = "/home/wuxiang/anaconda3/envs/torch20/lib/python3.9/site-packages/tgb/datasets/tgbn_"+name+"/ml_tgbn-"+name+"_node.pkl"
    if name == "reddit" or name == "token":
        
        root = "/home/wuxiang/anaconda3/envs/torch20/lib/python3.9/site-packages/tgb/datasets/tgbn_"+name
        nodefile = root + "/tgbn-" + name + "_node_labels.csv"
        OUT_NODE_DF = root + "/" + "ml_tgbn-{}_node.pkl".format(name)
        OUT_LABEL_DF = root + "/" + "ml_tgbn-{}_label.pkl".format(name)
        OUT_EDGE_FEAT = root + "/" + "ml_tgbn-{}.pkl".format(name + "_edge")
        node_ids = load_pkl(OUT_NODE_DF)
        labels_dict = load_pkl(OUT_LABEL_DF)
        node_label_dict = load_label_dict(
            nodefile, node_ids, labels_dict
        )
    else:
        node_label_dict = pd.read_pickle(path_nl)

    edge_index = read_npz(path_ei)
    edge_feature = read_npz(path_ef)

    # if name == 'token':
    #     for i in range(len(edge_feature)):
    #         edge_feature[i] = edge_feature[i]/1e10

    
    #node_feature = read_npz(path_nf)
    edge_time = read_npz(path_et)
    all_edge_time = np.hstack(edge_time)
    unique_elements, counts = np.unique(all_edge_time, return_counts=True)
    tims_span =(max(unique_elements) - min(unique_elements))/(3600*24)
    print("unique time:{}".format(len(counts)))
    nodes_num = int(np.max([np.max(arr) for arr in edge_index]))+1
    start_id = int(np.min([np.min(arr) for arr in edge_index]))+1
    #nodes_num = node_feature[0].shape[0]
    
    sub_graph = []
    for e_i in edge_index:
        row = e_i[0]
        col = e_i[1]
        ts = [1] * len(row)
        sub_g = coo_matrix((ts, (row, col)), shape=(nodes_num, nodes_num))
        sub_graph.append(sub_g)


    return sub_graph, edge_feature, edge_time,nodes_num,node_label_dict    



def read_npz(path):
    filesname = os.listdir(path)
    npz = []
    file_s = filesname.copy()
    for filename in filesname:
        id = filename.split('.')[0]
        id = int(id)
        file_s[id] = filename
    for filename in file_s:
        npz.append(np.load(path+filename))

    return npz


def read_e_feat(path):
    filesname = os.listdir(path)
    e_feat = []
    file_s = filesname.copy()
    for filename in filesname:
        id = filename.split('_')[0]
        id = int(id)
        file_s[id] = filename
    for filename in file_s:
        e_feat.append(np.load(path+filename))

    return e_feat


def read_graph(path, nodes_num, num):

    filesname = os.listdir(path)
    # 对文件名做一个排序
    file_s = filesname.copy()
    for filename in filesname:
        id = filename.split('_')[0]
        id = int(id) - num
        file_s[id] = filename

    # 文件读取
    sub_graph = []
    for file in file_s:
        sub_ = pd.read_csv(path + file)

        row = sub_.src_l.values
        col = sub_.dst_l.values

        node_m = set(row).union(set(col))
        # ts = torch.Tensor(sub_.timestamp.values)
        ts = [1] * len(row)

        sub_g = coo_matrix((ts, (row, col)), shape=(nodes_num, nodes_num))
        sub_graph.append(sub_g)

    return sub_graph


