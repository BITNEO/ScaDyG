from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import pandas as pd
import torch
import torch.nn as nn
from tqdm import trange
from sklearn import preprocessing
import torch.nn.functional as F
from model.loss import prediction, Link_loss_meta
from copy import deepcopy
import time
from transformer.Layers import EncoderLayer,EncoderLayer_with_scale 
import math
from model.utils import report_rank_based_eval_scalable
from torch_sparse import SparseTensor

from torch.nn import TransformerEncoderLayer, MultiheadAttention

from collections import defaultdict

import torch.nn.functional as F

def exp_rate(emb_seq,device,rate=1):
    length = emb_seq.shape[1]
    #distance = torch.tensor([i-length for i in range(1,length+1)])*rate
    #exp_rate = torch.exp(distance)
    exp_rate =torch.tensor([0.98**(length-i) for i in range(1,length+1)])
    exp_rate = exp_rate/exp_rate.sum()
    exp_rate = exp_rate.unsqueeze(0).unsqueeze(2).to(device)

    #exp_rate = torch.softmax(exp_rate,dim=0).unsqueeze(0).unsqueeze(2)
    # exp_rate = exp_rate.repeat(emb_seq.shape[0],1,1).to(device)
    
    emb_seq = exp_rate*emb_seq
    return emb_seq

def get_attn_key_pad_mask(seq_k, seq_q):
    ''' For masking out the padding part of key sequence. '''
    len_q = seq_q.size(1)
    # padding_mask = seq_k.eq(Constants.PAD)
    padding_mask = seq_k.eq(0)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk

    return padding_mask

def get_non_pad_mask(seq):
    assert seq.dim() == 2
    return seq.ne(0).type(torch.float).unsqueeze(-1)

class Encoder_trm(nn.Module):

    def __init__(self, device,len_max_seq, embed_dim_1,embed_dim_2, d_model, d_inner, n_layers, n_head,
                d_k, d_v, dropout=0.1):
        super().__init__()
        n_position = len_max_seq + 1
        self.device = device
        self.n_head = n_head

       
        self.transform_linear_time = nn.Linear(embed_dim_1,d_model).to(device)
        self.transform_linear_feature = nn.Linear(embed_dim_2,d_model).to(device)

        #self.weight_linear = nn.Linear(d_model,d_model).to(device)
        #self.bias_linear = nn.Linear(d_model,d_model).to(device)
        
        if n_layers == 1:
            self.layer_stack_time = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)]).to(device)
            self.layer_stack_feature = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)]).to(device)
        else:
            self.layer_stack_time = nn.ModuleList([
                EncoderLayer(d_model, d_model, n_head, d_k, d_v, dropout=dropout)]).to(device)
            #self.layer_stack_1.append(EncoderLayer_with_scale(d_model, d_inner, n_head, d_k, d_v, dropout=dropout))

            self.layer_stack_feature = nn.ModuleList([
                EncoderLayer_with_scale(d_model,d_inner, n_head, d_k, d_v, dropout=dropout)]).to(device)
            #self.layer_stack_2.append(EncoderLayer(d_model, d_model, n_head, d_k, d_v, dropout=dropout))
        #self.layer_stack.append(EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout))
    #@torch.no_grad()
    def forward(self, src_emb,src_emb_2, atten_mask=None, return_attns=True, needpos=False):
        src_emb = self.transform_linear_time(src_emb.float())
        src_emb_2 = self.transform_linear_feature(src_emb_2.float())
        enc_slf_attn_list = []
        src_pos_2 = torch.ones(src_emb_2.shape[0],src_emb_2.shape[1])
        #-- Prepare mask
        #print("atten_mask: ", atten_mask)
        if atten_mask == None:
            slf_attn_mask_2 = get_attn_key_pad_mask(seq_k=src_pos_2, seq_q=src_pos_2).to(self.device)
        else:
            slf_attn_mask_2 = atten_mask

        non_pad_mask_2 = get_non_pad_mask(src_pos_2).to(self.device)

        #-- Forward
        if needpos:
            enc_output_2 = src_emb_2 + self.position_enc(src_pos_2)
        else:
            enc_output_2 = src_emb_2.to(self.device)
        enc_output_2 = src_emb_2.to(self.device)

        src_pos_1 = torch.ones(src_emb.shape[0],src_emb.shape[1])
        #-- Prepare mask
        #print("atten_mask: ", atten_mask)
        if atten_mask == None:
            slf_attn_mask_1 = get_attn_key_pad_mask(seq_k=src_pos_1, seq_q=src_pos_1).to(self.device)
        else:
            slf_attn_mask_1 = atten_mask

        non_pad_mask_1 = get_non_pad_mask(src_pos_1).to(self.device)
        enc_output_1 = src_emb.to(self.device)
        # time
        for i,enc_layer in enumerate(self.layer_stack_time):
            if i == 0:
                enc_output_1, enc_slf_attn = enc_layer(
                    enc_output_1,non_pad_mask_1,slf_attn_mask_1)
                if return_attns:
                    enc_slf_attn_list += [enc_slf_attn]
            else:
                enc_output_1, enc_slf_attn = enc_layer(
                    enc_output_1,non_pad_mask_1,slf_attn_mask_1)
                if return_attns:
                    enc_slf_attn_list += [enc_slf_attn]

        # weight_scale = self.weight_linear(enc_output_2) 
        # weight_bias = self.bias_linear(enc_output_2)
        

        
        for i,enc_layer in enumerate(self.layer_stack_feature):
            if i==0:
                enc_output_2, enc_slf_attn = enc_layer(
                    enc_output_2,enc_output_1,non_pad_mask_2,slf_attn_mask_2)
                if return_attns:
                    enc_slf_attn_list += [enc_slf_attn]
            else:
                enc_output_2, enc_slf_attn = enc_layer(
                    enc_output_2,enc_output_1,non_pad_mask_2,slf_attn_mask_2)
                if return_attns:
                    enc_slf_attn_list += [enc_slf_attn]
        
        if return_attns:
            
            #enc_slf_attn = torch.softmax(enc_slf_attn_list[-1],dim=1)
            enc_output = torch.softmax(enc_output_2,dim=1)
            return enc_output,enc_slf_attn
        #enc_output = torch.softmax(enc_output)
        else:
            return enc_slf_attn_list[0]


class CustomEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=100, dropout=0.1):
        super(CustomEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)

        # Note: Norm and Dropout layers can be included or excluded as per requirement
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.ffn = nn.Linear(d_model, dim_feedforward)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # Self attention
        #seq_length = src.shape[1]

        src = self.self_attn(src, src, src)[0]
        
        src = self.norm1(src)
        src = self.ffn(src)
        

        
        return src


        
        

        


class CustomEncoderLayer_withScale(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=100, dropout=0.1):
        super(CustomEncoderLayer_withScale, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)

        # Note: Norm and Dropout layers can be included or excluded as per requirement
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.ffn = nn.Linear(d_model, dim_feedforward)

    def forward(self, src,src_1 ,src_mask=None, src_key_padding_mask=None):
        # Self attention
        src = self.self_attn(src, src, src)[0]
        src =  self.dropout1(src)
        src = self.norm1(src)
        src = src+src_1
        src = self.ffn(src)
        
   
        
        return src
    
class Encoder(nn.Module):

    def __init__(self, device, embed_dim_1,embed_dim_2, d_model, d_inner, n_layers, n_head,
                d_k, d_v, dropout=0.1):
        super().__init__()
        self.device = device
        self.n_head = n_head
        self.sp = torch.nn.Softplus()
        self.sigmoid = torch.nn.Sigmoid()
        self.d_inner = d_inner

       
        self.transform_linear_time = nn.Linear(embed_dim_1,embed_dim_1).to(device)
        self.transform_linear_feature = nn.Linear(embed_dim_1,d_model).to(device)
        #self.weight = nn.Parameter(torch.Tensor(embed_dim_1, embed_dim_1))
        self.weight = nn.Parameter(torch.Tensor(embed_dim_1, embed_dim_1))
        self.scale_matrix = nn.Parameter(torch.Tensor(embed_dim_1,embed_dim_1))
        self.global_weight = nn.Parameter(torch.Tensor(embed_dim_1))
        self.bias = nn.Parameter(torch.Tensor(embed_dim_1))
        nn.init.constant_(self.bias, 0)


        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.scale_matrix, a=math.sqrt(5))
        fan_in = self.scale_matrix.size(0)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.global_weight, -bound, bound)


        if d_inner <= 4:
            self.layer_stack_time_1 = CustomEncoderLayer(d_model,n_head, d_inner, dropout=dropout).to(device)
            self.layer_stack_time_2 = CustomEncoderLayer(d_model,n_head, d_inner, dropout=dropout).to(device)
            self.layer_stack_feature = CustomEncoderLayer_withScale(d_model, n_head,d_model, dropout=dropout).to(device)
        else:
            self.layer_stack_time_1 = CustomEncoderLayer(d_model,n_head, 4**2, dropout=dropout).to(device)
    
    def repeat_to_n_dim(self,matrix, n):
       
        repeat_times = -(-n // 16)
        
       
        repeated_matrix = matrix.repeat(1, 1, repeat_times)
        result_matrix = repeated_matrix[:, :, :n]
        
        return result_matrix



        
    def forward(self, src_emb,src_emb_2,mode = 'train',step1 = 0,step2=0):# time feature, node feature  src_emb:[node_num,seq_len,node_feat_dim]
        # src_emb_feat = self.transform_linear_feature(src_emb)
        # enc_output_1 = self.layer_stack_time_1(src_emb_feat)

        # one matrix for all
        # src_emb = torch.sum(src_emb,dim=1)
        # src_emb = src_emb
        # result = torch.matmul(src_emb,self.weight )



        # a matrix generate a size*size matrix then reshape
        # src_emb = torch.sum(src_emb,dim=1)
        # result = torch.matmul(src_emb,self.weight)
        # result = result.reshape(result.shape[0],src_emb.shape[1],src_emb.shape[1])
        
        # src_emb = src_emb.unsqueeze(-1)

        # src_emb = torch.bmm(result,src_emb)
        # result = src_emb.squeeze(-1)


        # scaling
        # src_emb = torch.sum(src_emb,dim=1)
        # scale = self.sigmoid(self.scale_weight*src_emb)
        # new_weight = scale.unsqueeze(-1).repeat(1,1,src_emb.shape[-1])
        # new_weight = self.weight * new_weight
        # src_emb = src_emb.unsqueeze(-1)
        # result =  torch.bmm(new_weight,src_emb)
        # result = result.squeeze(-1)

        #scaling 2d
        src_emb = torch.sum(src_emb,dim=1)
        src_emb_1 = src_emb.unsqueeze(-1)
        scale_weight = self.scale_matrix.unsqueeze(0).repeat(src_emb.shape[0],1,1)
        scale = torch.bmm(scale_weight,src_emb_1)
        

        
        global_weight = self.global_weight.unsqueeze(0).expand(src_emb.shape[0], -1, -1)
        new_weight = self.sigmoid(torch.bmm(scale,global_weight ))

        if mode == "val" and step2==0:
            torch.save(new_weight, './weights/tensor_file'+str(step1)+'.pth')
        torch.save(self.weight, './weights/self_weight.pth')

        #new_weight = scale.unsqueeze(-1).repeat(1,1,src_emb.shape[-1])
        new_weight = self.weight * new_weight
        #new_weight = new_weight.unsqueeze(2)
        src_emb = src_emb.unsqueeze(-1)
        result =  torch.bmm(new_weight,src_emb)
        result = result.squeeze(-1)




        # src_emb_feat = self.transform_linear_feature(src_emb)
        # enc_output_1 = self.layer_stack_time_1(src_emb_feat)
        # #weight = self.sp(self.weight)
        # #weight = self.weight.view(1, 1, src_emb.shape[-1], src_emb.shape[-1])
        # # enc_output_1 = enc_output_1.unsqueeze(-1)
        # # weight = self.sp(enc_output_1*weight)
        # #src_emb = src_emb.unsqueeze(-1)
        # result = torch.matmul(src_emb, self.weight).squeeze(-1) 

        
        # src_emb_feat = self.transform_linear_feature(src_emb)
        # enc_output_1 = self.layer_stack_time_1(src_emb_feat)
        # enc_output_1 = torch.softmax(enc_output_1,dim=1)
        # if self.d_inner > 4:
        #     enc_output_1 = self.repeat_to_n_dim(enc_output_1,self.d_inner**2 )
        # enc_output_1 = enc_output_1.reshape(enc_output_1.shape[0],enc_output_1.shape[1],src_emb.shape[-1],src_emb.shape[-1])
        
        # src_emb = src_emb.unsqueeze(-1)
        # result = torch.matmul(enc_output_1, src_emb).squeeze(-1)

        

        

        


        return result

class TimeEncode_exp(nn.Module):
    def __init__(self, dim, args):
        super(TimeEncode_exp, self).__init__()
        self.dim = dim
        self.a = args.a
        self.base_value = args.base_value
        self.w = nn.Linear(1, dim)
        self.reset_parameters()
    
    def reset_parameters(self, ):
        #mask_vector = np.tile([1, -1], 50)
        vec = torch.linspace(-self.a , -self.a *0.1, self.dim)
        #vec =torch.from_numpy(-(1/10** np.linspace(0, self.a, self.dim, dtype=np.float32)))
        #vec = torch.Tensor([-0.000001,-0.00001])
        self.w.weight = nn.Parameter(vec.unsqueeze(1).float())
       
        #self.w.weight = nn.Parameter(torch.rand(100).unsqueeze(1))

        # self.w.bias = nn.Parameter(torch.zeros(self.dim))
        #self.w.weight = torch.tensor(-1)
        self.w.bias = nn.Parameter(torch.zeros(self.dim))

        self.w.weight.requires_grad = False
        self.w.bias.requires_grad = False
    
    #@torch.no_grad()
    def forward(self, t):
        output = torch.exp(self.w(t.reshape((-1, 1))))
        return output
    
class TimeEncode(nn.Module):
    """
    out = linear(time_scatter): 1-->time_dims
    out = cos(out)
    """
    def __init__(self, dim, a=100):
        super(TimeEncode, self).__init__()
        self.dim = dim
        self.a = a
        self.w = nn.Linear(1, dim)
        self.reset_parameters()
    
    def reset_parameters(self, ):
        self.w.weight = nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, self.a, self.dim, dtype=np.float32))).reshape(self.dim, -1))
        self.w.bias = nn.Parameter(torch.zeros(self.dim))

        self.w.weight.requires_grad = False
        self.w.bias.requires_grad = False
    
    #@torch.no_grad()
    def forward(self, t):
        output = torch.cos(self.w(t.reshape((-1, 1))))
        return output



class snapshot():
    def __init__(self, time, graph, args, device):
        self.d = 60
        self.graph = graph
        self.time = time
        self.node_num = graph.num_nodes()
        self.time_enc = TimeEncode_exp(graph.edge_feature.shape[1], args).to(device)
        #self.snapshot_timefeat = self.time_enc(torch.tensor(0).to(torch.float32).to(device))
        self.device = device
        
        #Initialize `edge_list` as the index of a sparse matrix.
        edge_index = torch.stack(graph.edges()).t().to(device)
        self.initialize_edge_node_mat(edge_index)
    
    def initialize_edge_node_mat(self, edge_index):
        self.time_features = self.time_enc(self.time- self.graph.edge_time.to(torch.float32).to(self.device))
        #self.time_rate = torch.matmul(self.time_features, self.snapshot_timefeat.t()) 
        self.time_rate = torch.ones(edge_index.shape[0]).to(self.device)
        
        self.adj_matrix = self.graph.adjacency_matrix().to(self.device)
        
        
        in_degree = 1 / self.graph.in_degrees().to(torch.float).to(self.device)
        in_degree[in_degree == float('inf')] = 0
        row_indices = torch.arange(self.node_num, device=self.device)
        col_indices = torch.arange(self.node_num, device=self.device)
        degree_matrix = SparseTensor(
            row=row_indices, col=col_indices,
            value=in_degree, sparse_sizes=(self.node_num, self.node_num)
        )        
        self.inverse_degree_matrix = degree_matrix
        

        
        edge_values = self.time_rate.repeat(2) 
        node_indices = torch.cat([edge_index[:,0], edge_index[:,1]]) 
        edge_indices = torch.arange(0, edge_index.size(0)).repeat(2).to(self.device)  
        self.node_edge_mat = SparseTensor(row=node_indices, col=edge_indices, value=edge_values, sparse_sizes=(self.node_num, edge_index.size(0)))
        self.edge_node_mat = self.node_edge_mat.t()
        
    def set_edge_features(self, feature):
        feature = feature.to(self.device)
        feature =feature*self.time_features
        node_feature = self.node_edge_mat.matmul(feature, reduce='sum')  # 使用 mean 或者 sum 作为 reduce 方式
        feature = feature.cpu()
        return node_feature.to(torch.device("cpu"))
    
    def set_node_features(self, node_feature):
        node_feature = node_feature.to(self.device)

        #result_feature = torch.sparse.mm(self.edge_node_mat, node_feature)
        result_feature = self.edge_node_mat.matmul( node_feature,reduce='sum')
        result_feature = self.node_edge_mat.matmul(result_feature, reduce='sum')
        return result_feature.to(torch.device("cpu"))
    

    def empty(self):
       
        del self.graph
        #del self.snapshot_timefeat
        del self.adj_matrix
        del self.node_edge_mat
        del self.time_features
        del self.inverse_degree_matrix 
        del self.time_rate
        

        
        import gc
        gc.collect()

        
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()





def graph_coarsening(node_feature,coarsening_num):
    node_num = node_feature.shape[0]
    new_node_num = math.floor(node_num/coarsening_num)
    coarsened_feature = [] 
    coarsed_node_feature = []
    for n in range(new_node_num):
        for i in range(coarsening_num):
            coarsed_node_feature.append(node_feature[new_node_num*i+n])
        coarsened_feature.append(torch.mean(torch.stack(coarsed_node_feature),dim=0))
        coarsed_node_feature = []

    
            
    if node_num%coarsening_num != 0:
        coarsened_feature.append(torch.mean(node_feature[new_node_num*coarsening_num:],dim=0))
    return torch.stack(coarsened_feature)

def weight_expand(weight,coarsening_num,node_num):
    if node_num % coarsening_num == 0:
        weight_expand = weight.repeat(coarsening_num,1)
    else:
        weight_1 = weight[-1]
        if len(weight_1.shape)==2:
            weight_1 = weight_1.unsqueeze(0)
        weight_expand = torch.cat([weight[:-1].repeat(coarsening_num,1,1),weight_1.repeat(node_num%coarsening_num,1,1)],dim=0)
    return weight_expand

            

def data_partition(data_1,data_2,partition_num):
    data_1 = torch.chunk(data_1, partition_num)
    data_2 = torch.chunk(data_2,partition_num)
    return data_1,data_2

def interleave_even_list(lst):
    
    mid = len(lst) // 2
    
   
    first_half = lst[:mid]
    second_half = lst[mid:]
    
   
    interleaved_list = []
    for i in range(mid):
        interleaved_list.append(first_half[i])
        interleaved_list.append(second_half[i])
    
    return interleaved_list

class Predict_layer(nn.Module):
    def __init__(self,input_dim1,hidden_dim,output_dim,hop):
        """
        Customized dataset.
        :param indices_list: list, list of indices
        """

        
        super(Predict_layer, self).__init__()
        if hop == 1:
            self.fc1 = nn.Linear(input_dim1*2, hidden_dim)
        elif hop == 2:
            self.fc1 = nn.Linear(input_dim1*4, hidden_dim)
        else:
            self.fc1 = nn.Linear(input_dim1*6, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.act = nn.ReLU()
      
    
    def forward(self,edge_index,x,mode='train'):

        if mode == 'train':
            
            node_feat = x[edge_index]
            nodes_1 = node_feat[0]
            nodes_2 = node_feat[1]
            x = torch.cat([nodes_1, nodes_2], dim=1)
            #x = torch.cat([nodes_1, nodes_2], dim=1)
            h = torch.sigmoid(self.fc2(self.act(self.fc1(x))))
        else:
            node_feat = x[edge_index.edge_label_index]
            nodes_1 = node_feat[0]
            nodes_2 = node_feat[1]
            x = torch.cat([nodes_1, nodes_2], dim=1)
            h = torch.sigmoid(self.fc2(self.act(self.fc1(x))))
        return h


class Predict_layer_1(nn.Module):
    def __init__(self,input_dim1,hidden_dim,output_dim,hop):
        """
        Customized dataset.
        :param indices_list: list, list of indices
        """

        
        super(Predict_layer_1, self).__init__()
        if hop == 1:
            self.fc1 = nn.Linear(input_dim1, hidden_dim)
        elif hop == 2:
            self.fc1 = nn.Linear(input_dim1*4, hidden_dim)
        else:
            self.fc1 = nn.Linear(input_dim1*6, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.act = nn.ReLU()
        self.decode_module =nn.CosineSimilarity(dim=-1)
        self.dropout = 0.1
    
    def forward(self,edge_index,x,mode='train'):

        if mode == 'train':
            node_feat = x[edge_index]
            nodes_1 = self.act(self.fc1(node_feat[0]))
            nodes_2 = self.act(self.fc1(node_feat[1]))
            #x = torch.cat([nodes_1, nodes_2], dim=1)
            x = self.decode_module(nodes_1,nodes_2)
            h = torch.sigmoid(x)
            #x = torch.cat([nodes_1, nodes_2], dim=1)
        else:
            node_feat = x[edge_index.edge_label_index]
            nodes_1 = self.fc1(node_feat[0])
            nodes_2 = self.fc1(node_feat[1])
            #x = torch.cat([nodes_1, nodes_2], dim=1)
            x = self.decode_module(nodes_1,nodes_2)
            h = torch.sigmoid(x)
            #x = torch.cat([nodes_1, nodes_2], dim=1)
        return h.unsqueeze(1)
    
 

class merge_weight(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(merge_weight, self).__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        
    def forward(self, x1,x2):
        x = self.fc1(torch.cat([x1,x2],dim=1))
        return torch.softmax(x,dim=1)

def graph_coarsening(node_feature,coarsening_num):
    node_num = node_feature.shape[0]
    new_node_num = math.floor(node_num/coarsening_num)
    coarsened_feature = [] 
    coarsed_node_feature = []
    for n in range(new_node_num):
        for i in range(coarsening_num):
            coarsed_node_feature.append(node_feature[new_node_num*i+n])
        coarsened_feature.append(torch.mean(torch.stack(coarsed_node_feature),dim=0))
        coarsed_node_feature = []

    
            
    if node_num%coarsening_num != 0:
        coarsened_feature.append(torch.mean(node_feature[new_node_num*coarsening_num:],dim=0))
    return torch.stack(coarsened_feature)

def weight_expand(weight,coarsening_num,node_num):
    if node_num % coarsening_num == 0:
        weight_expand = weight.repeat(coarsening_num,1)
    else:
        weight_1 = weight[-1]
        if len(weight_1.shape)==2:
            weight_1 = weight_1.unsqueeze(0)
        weight_expand = torch.cat([weight[:-1].repeat(coarsening_num,1,1),weight_1.repeat(node_num%coarsening_num,1,1)],dim=0)
    return weight_expand

            
            

            


def train_scalable_tgn(model, model_transformer,optimizer, device, graph_l, logger, train_n,val_n,test_n,args):
    best_param = {'best_acc': 0, 'best_state': None, 'best_s_dw': None}
    graph_train = graph_l[:train_n]
    graph_val = graph_l[train_n:val_n]
    graph_test = graph_l[val_n:]
    print("train steps: {}, val steps {}, test steps {} ".format(len(graph_train),len(graph_val),len(graph_test)))
    hop = 3
        #last_timestamp = 0
    last_snapshot = 0
    last_node_feature = 0
    last_two_hop_feature = 0
    last_three_hop_feature = 0
    train_feature_list = []
    val_feature_list = []
    test_feature_list = []

    train_feature_list_coarsened = []
    val_feature_list_coarsened = []
    test_feature_list_coarsened = []

    train_2hop_feature_list = []
    val_2hop_feature_list = []
    test_2hop_feature_list = []
    train_3hop_feature_list = []
    val_3hop_feature_list = []
    test_3hop_feature_list = []
    snapshot_timestamp_list = []
    start_time = time.time()
    

    for idx, graph in enumerate(graph_train):
        print(idx)
        snapshot_timestamp = graph.edge_time.max()
        snapshot_i = snapshot(snapshot_timestamp,graph,args,device=device)
        snapshot_timestamp_list.append(snapshot_timestamp)
        feature = snapshot_i.set_edge_features(graph.edge_feature).to(device)
        node_feature = snapshot_i.set_node_features(feature).to(device)
        if args.dataset in  ['reddit_title','USLegis','UNovte','SocialEvo','Flights','enron']:
            node_feature = snapshot_i.set_node_features(graph.node_feature).to(device)
            feature = torch.cat([feature,node_feature],dim=1)

        snapshot_i.empty()

        train_feature_list.append(feature.cpu())

        last_node_feature= feature.cpu()

   
    
    for idx, graph in enumerate(graph_val):
        print(idx)
        snapshot_timestamp = graph.edge_time.max()
        snapshot_i = snapshot(snapshot_timestamp,graph,args,device=device)
        snapshot_timestamp_list.append(snapshot_timestamp)
        feature = snapshot_i.set_edge_features(graph.edge_feature).to(device)
        if args.dataset in  ['reddit_title','USLegis','UNovte','SocialEvo','Flights','enron']:
            node_feature = snapshot_i.set_node_features(graph.node_feature).to(device)
            feature = torch.cat([feature,node_feature],dim=1)
        snapshot_i.empty()
        
        val_feature_list.append(feature.cpu())
        
 
        last_node_feature= feature.cpu()
        last_snapshot = snapshot_i
        # last_two_hop_feature= two_hop_feature
        # last_three_hop_feature = three_hop_feature
    
    for idx, graph in enumerate(graph_test):
        print(idx)
        snapshot_timestamp = graph.edge_time.max()
        snapshot_i = snapshot(snapshot_timestamp,graph,args,device=device)
        snapshot_timestamp_list.append(snapshot_timestamp)
        feature = snapshot_i.set_edge_features(graph.edge_feature).to(device)
        if args.dataset in  ['reddit_title','USLegis','UNovte','SocialEvo','Flights','enron']:
            node_feature = snapshot_i.set_node_features(graph.node_feature).to(device)
            feature = torch.cat([feature,node_feature],dim=1)
     
        snapshot_i.empty()
      
        test_feature_list.append(feature.cpu())
        last_node_feature= feature.cpu()
        last_snapshot = snapshot_i
     
    train_feature_list = torch.stack(train_feature_list)
    val_feature_list = torch.stack(val_feature_list)
    test_feature_list = torch.stack(test_feature_list)

    time_enc = TimeEncode_exp(train_feature_list.shape[2],args)

    end_time = time.time()
    print("elapsed time:"+str(end_time-start_time)+"s")
    n = args.n

    # with historical states
    snapshot_timestamp_list = torch.stack(snapshot_timestamp_list)
    for epoch in range(args.epochs):
        st_time = time.time()
        for idx in range(len(graph_train)-1):

            if idx>n:
                time_feat = time_enc(snapshot_timestamp_list[idx]-snapshot_timestamp_list[idx-n:idx+1])
                input_timefeat_list = time_feat.unsqueeze(0).to(device)
                hist_node_feature_list = train_feature_list[idx-n:idx+1].permute(1,0,2)
                
                his_state = torch.mean(train_feature_list[:idx-n],dim=0).unsqueeze(0).permute(1,0,2)
                #hist_node_feature_list = torch.cat([his_state,hist_node_feature_list],dim=1)
                
            else:
                time_feat = time_enc(snapshot_timestamp_list[idx]-snapshot_timestamp_list[:idx+1])

                input_timefeat_list = time_feat.unsqueeze(0).to(device)
                hist_node_feature_list = train_feature_list[:idx+1].permute(1,0,2)
            
            hist_node_feature_list = hist_node_feature_list*time_feat
            edge_index = graph_train[idx+1].edge_label_index.t()
            edge_label= graph_train[idx+1].edge_label.unsqueeze(1)
            edge = torch.cat([edge_index,edge_label],dim=1).long()
            
            indices = torch.randperm(len(edge))

            edge = edge[indices]
            

            
            
            
            if edge.shape[0] > args.batch_size:
                
                
                edge = torch.chunk(edge,math.floor(edge.shape[0]/args.batch_size))
            else:
                edge = [edge]

            for batch in edge:
                edge_index_i = batch[:,:2]
                edge_label_i = batch[:,2]

                

                node_list = list(torch.unique(torch.flatten(edge_index_i)))
                node_list = [int(node) for node in node_list]
                
                element_to_rank = {element: rank for rank,element in enumerate(node_list)}
                edge_index_reorder = [torch.tensor([element_to_rank[element[0].item()],element_to_rank[element[1].item()]]) for element in edge_index_i]
                edge_index_reorder = torch.stack(edge_index_reorder)
                edge_index_reorder = edge_index_reorder.t()


                
                
                



                batch_hist_node_feature_list = hist_node_feature_list[node_list].clone()



                batch_hist_node_feature_list = batch_hist_node_feature_list.to(device)
                batch_hist_node_feature_list = model_transformer(batch_hist_node_feature_list,batch_hist_node_feature_list)
                
                
            
                #feature = weight*batch_hist_node_feature_list
                #feature = exp_rate(batch_hist_node_feature_list,device=device,rate=args.time_rate)
                #batch_hist_node_feature_list = torch.sum(batch_hist_node_feature_list, dim=1)

                #feature = feature[edge_index_reorder]
            
            
            
                pred = model(edge_index_reorder,batch_hist_node_feature_list,mode = 'train'
                                ).squeeze(1)
                loss = Link_loss_meta(pred, edge_label_i)
                
            
            
            

            #if losses:
                #losses = losses / count
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                feature =feature.cpu()
                
                #logger.info('meta epoch:{},  loss: {:.5f}'.format(epoch,  losses))
        ed_time = time.time()
        print(ed_time-st_time)
        val_acc = 0
        count = 0
        val_ap =0
        with torch.no_grad():
            st_time = time.time()
            for idx in range(len(graph_val)):
                if idx >= n:
                    hist_node_feature_list = val_feature_list[idx-n:idx].permute(1,0,2).clone().to(device)

                    input_timefeat = time_enc(snapshot_timestamp_list[idx+len(graph_train)-1]-snapshot_timestamp_list[idx+len(graph_train)-n:idx+len(graph_train)]).unsqueeze(0).to(device)
                else:
                    # 原始 是 len*nodenum*featdim
                    if n >= idx+len(graph_train):
                        input_timefeat = time_enc(snapshot_timestamp_list[idx+len(graph_train)-1]-snapshot_timestamp_list[:idx+len(graph_train)]).unsqueeze(0).to(device)
                        hist_node_feature_list = torch.cat([train_feature_list.clone(),val_feature_list[:idx].clone()],dim=0).permute(1,0,2).to(device)
                    else:

                        hist_node_feature_list = torch.cat([train_feature_list[-(n-idx):].clone(),val_feature_list[:idx].clone()],dim=0).permute(1,0,2).to(device)
                        #hist_node_feature_list_coarsened = train_feature_list_coarsened[-n:].permute(1,0,2)
                        input_timefeat = time_enc(snapshot_timestamp_list[idx+len(graph_train)-1]-snapshot_timestamp_list[idx+len(graph_train)-n:idx+len(graph_train)]).unsqueeze(0).to(device)
                        #his_state = torch.mean(train_feature_list[:idx-n],dim=0).unsqueeze(0).permute(1,0,2)
                    #hist_node_feature_list = torch.cat([his_state,hist_node_feature_list],dim=1)
                if args.hop>=2:
                    hist_node_feature_list_2hop = train_2hop_feature_list[:idx+1].permute(1,0,2)
                if args.hop>=3:
                    hist_node_feature_list_3hop = train_3hop_feature_list[:idx+1].permute(1,0,2)
                
                hist_node_feature_list = hist_node_feature_list*input_timefeat
                all_feature = []
                if hist_node_feature_list.shape[0]>args.batch_size:
                    hist_node_feature_list = torch.chunk(hist_node_feature_list,math.floor(hist_node_feature_list.shape[0]/args.batch_size))
                else:
                    hist_node_feature_list = hist_node_feature_list.unsqueeze(1)
                zero_number = 0
                
                for i,batch_node in enumerate(hist_node_feature_list):
                    batch_node = batch_node.to(device)
                    #zero_elements = torch.eq(batch_node, 0)

                    # 使用torch.nonzero()获取值为0的元素的索引
                    # zero_indices = torch.nonzero(zero_elements)
                    # zero_number += (len(zero_indices)/batch_node.shape[2])
                    
                    batch_node = model_transformer(batch_node,batch_node)
                    

                
                    # weight = output[:,:,0].unsqueeze(2)
                    # feature = weight*batch_node
                    #feature = exp_rate(batch_node,device=device,rate=args.time_rate)

                    #batch_node = torch.sum(batch_node, dim=1)
                    #feature = iterative_add(weight,hist_node_feature_list)
                    all_feature.append(batch_node)
                feature = torch.cat(all_feature,dim=0)
                # print(zero_number/feature.shape[0])
                
                
                
                graph_val[idx] = graph_val[idx].to(device)
                feature = feature.to(device)
                model = model.to(device)
                pred = model(graph_val[idx],feature,mode='val').squeeze(1)
                loss = Link_loss_meta(pred, graph_val[idx].edge_label)
                edge_label = graph_val[idx].edge_label
                edge_label_index = graph_val[idx].edge_label_index
                
                if args.dataset in ['UNvote']:
                    mrr, rl1, rl3, rl10 = report_rank_based_eval_scalable(model, graph_val[idx], feature,num_neg_per_node=50)
                else:
                    mrr, rl1, rl3, rl10 = report_rank_based_eval_scalable(model.cpu(), graph_val[idx].cpu(), feature.cpu(),num_neg_per_node=100)
                graph_val[idx].edge_label = edge_label
                graph_val[idx].edge_label_index = edge_label_index
                acc, ap, f1, macro_auc, micro_auc = prediction(pred, graph_val[idx].edge_label)
                

                
                logger.info('meta val epoch:{}, loss: {:.5f},mrr:{:.5f},  acc: {:.5f}, ap: {:.5f}, f1: {:.5f}, macro_auc: {:.5f}, micro_auc: {:.5f}'.
                                format(epoch,  loss, mrr,acc, ap, f1, macro_auc, micro_auc))
                feature =feature.cpu()
                model = model.to(device)
                val_acc += acc
                val_ap += ap
                count +=1
            ed_time = time.time()
            print(ed_time-st_time)
        print(count)
        val_acc = val_acc / count
        val_ap = val_ap / count
        if val_ap > best_param['best_acc']:
            best_param = {'best_acc': val_ap, 'best_state': deepcopy(model.state_dict())}
            earl_stop_c = 0
        else:
            earl_stop_c += 1
            if earl_stop_c == 10:
                break
    # testing

    model.load_state_dict(best_param['best_state'])
    model.eval()
    avg_mrr = 0.0
    avg_acc = 0.0
    avg_auc = 0.0
    avg_rcall10 = 0.0
    avg_ap = 0.0
    len_test = len(graph_test)
    all_edge = 0
    with torch.no_grad():
        for idx in range(len(graph_test)):

            if idx == 24:
                print(1)
            
            total_past_len = len(graph_train) + len(graph_val)
            total_past_feat = torch.cat([train_feature_list,val_feature_list],dim=0)
            if n >= idx+(len(graph_train)+len(graph_val)):
                input_timefeat = time_enc(snapshot_timestamp_list[idx + total_past_len-1]-snapshot_timestamp_list[: idx + total_past_len]).unsqueeze(0).to(device)
            else:
                input_timefeat = time_enc(snapshot_timestamp_list[ idx + total_past_len-1]-snapshot_timestamp_list[idx + total_past_len - n: idx + total_past_len]).unsqueeze(0).to(device)
            

            
            if idx >= n:
                hist_node_feature_list = test_feature_list[idx-n:idx]
                his_state = torch.cat([total_past_feat,test_feature_list[:idx-n]])
                his_state = torch.mean(his_state,dim=0).unsqueeze(0)
                #hist_node_feature_list = torch.cat([his_state,hist_node_feature_list],dim=0)
                #hist_node_feature_list_coarsened = test_feature_list_coarsened[idx-n:idx].clone()
                if args.hop>=2:
                    hist_node_feature_list_2hop = train_2hop_feature_list[:idx+1].permute(1,0,2).to(device)
                if args.hop>=3:
                    hist_node_feature_list_3hop = train_3hop_feature_list[:idx+1].permute(1,0,2).to(device)
                
            else:
                if n >= idx+(len(graph_train)+len(graph_val)):
                    hist_node_feature_list = torch.cat([total_past_feat,test_feature_list[:idx]],dim=0).clone()
                    
                hist_node_feature_list= torch.cat([total_past_feat[-(n-idx):],test_feature_list[:idx]],dim=0).clone()
                
                #his_state = torch.mean(total_past_feat[:-n],dim=0).unsqueeze(0)
                #hist_node_feature_list = torch.cat([his_state,hist_node_feature_list],dim=0)
                if args.hop>=2:
                    hist_node_feature_list_2hop = train_2hop_feature_list[:idx+1].permute(1,0,2).to(device)
                if args.hop>=3:
                    hist_node_feature_list_3hop = train_3hop_feature_list[:idx+1].permute(1,0,2).to(device)
            print(idx)
            hist_node_feature_list = hist_node_feature_list.permute(1,0,2).to(device)
            hist_node_feature_list = hist_node_feature_list*input_timefeat
            all_feature = []
            if hist_node_feature_list.shape[0]>args.batch_size:
                hist_node_feature_list = torch.chunk(hist_node_feature_list,math.floor(hist_node_feature_list.shape[0]/args.batch_size))
            else:
                hist_node_feature_list = hist_node_feature_list.unsqueeze(1)
            for i,batch_node in enumerate(hist_node_feature_list):
                batch_node = batch_node.to(device)
                batch_node = model_transformer(batch_node,batch_node,mode = 'val',step1 = idx,step2=i)
                # weight = output[:,:,0].unsqueeze(2)
                # feature = weight*batch_node
                #feature = exp_rate(batch_node,device=device,rate=args.time_rate)
                #batch_node = torch.sum(batch_node, dim=1)
                all_feature.append(batch_node)
            all_feature = torch.cat(all_feature,dim=0)
            
            model = model.to(device)
            all_feature = all_feature.to(device)
            graph_test[idx] = graph_test[idx].to(device)
            pred = model(graph_test[idx],all_feature,mode='val').squeeze(1)
            loss = Link_loss_meta(pred, graph_test[idx].edge_label)
            edge_label = graph_test[idx].edge_label
            edge_label_index = graph_test[idx].edge_label_index
            mrr, rl1, rl3, rl10 = report_rank_based_eval_scalable(model.cpu(), graph_test[idx].cpu(), all_feature.cpu(),num_neg_per_node=100)
            
            graph_test[idx].edge_label = edge_label
            graph_test[idx].edge_label_index = edge_label_index
            acc, ap, f1, macro_auc, micro_auc = prediction(pred, graph_test[idx].edge_label)
            logger.info('test epoch:{}, loss: {:.5f},mrr:{:.5f}, acc: {:.5f}, ap: {:.5f}, f1: {:.5f}, macro_auc: {:.5f}, micro_auc: {:.5f}'.
                            format(epoch,  loss, mrr,acc, ap, f1, macro_auc, micro_auc))
            feature =feature.cpu()
            del all_feature
            num_edges = graph_test[idx].num_edges()
            avg_mrr += mrr*num_edges
            avg_ap += ap*num_edges
            avg_acc += acc*num_edges
            avg_auc += macro_auc*num_edges
            avg_rcall10 += rl10*num_edges
            all_edge +=num_edges
        
    #print("sum"+str(i))
    avg_mrr /= all_edge
    avg_acc /= all_edge
    avg_auc /= all_edge
    avg_rcall10 /= all_edge
    avg_ap /= all_edge
    logger.info({'avg_mrr': avg_mrr})
    logger.info({'avg_acc': avg_acc})
    logger.info({'avg_auc': avg_auc})
    logger.info({'avg_rcall10': avg_rcall10})
    logger.info({'avg_ap': avg_ap})
    return avg_mrr,avg_auc,avg_acc
    
            
    


    
    

       
