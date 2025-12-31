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
from transformer.Layers import EncoderLayer     

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

class Encoder(nn.Module):

    def __init__(self, device,len_max_seq, embed_dim, d_model, d_inner, n_layers, n_head,
                d_k, d_v, dropout=0.1):
        super().__init__()
        n_position = len_max_seq + 1
        self.device = device
        self.n_head = n_head

       
        self.transform_linear = nn.Linear(embed_dim,d_model).to(device)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_model, n_head, d_k, d_v, dropout=dropout)]).to(device)
        self.layer_stack.append(EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout))
        #self.layer_stack.append(EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout))
    #@torch.no_grad()
    def forward(self, src_emb, atten_mask=None, return_attns=True, needpos=False):
        src_emb = self.transform_linear(src_emb.float())
        enc_slf_attn_list = []
        src_pos = torch.ones(src_emb.shape[0],src_emb.shape[1])
        #src_pos = torch.linspace(0,src_emb.shape[1], src_emb.shape[1]).unsqueeze(0).repeat(src_emb.shape[0],src_emb.s)
        #-- Prepare mask
        #print("atten_mask: ", atten_mask)
        if atten_mask == None:
            slf_attn_mask = get_attn_key_pad_mask(seq_k=src_pos, seq_q=src_pos).to(self.device)
        else:
            slf_attn_mask = atten_mask

        non_pad_mask = get_non_pad_mask(src_pos).to(self.device)

        #-- Forward
        if needpos:
            enc_output = src_emb + self.position_enc(src_pos)
        else:
            enc_output = src_emb.to(self.device)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output,non_pad_mask,slf_attn_mask)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        if return_attns:
            
            #enc_slf_attn = torch.softmax(enc_slf_attn_list[-1],dim=1)
            enc_slf_attn = torch.softmax(enc_slf_attn_list[-1],dim=1)
            tensor_list = torch.chunk(enc_slf_attn, chunks=8, dim=0)
            tensor_list = torch.mean(torch.stack(tensor_list), dim=0)
            #enc_output = torch.softmax(enc_output,dim=1)
            return enc_output,tensor_list
        #enc_output = torch.softmax(enc_output)
        else:
            return enc_slf_attn_list[0]

class TimeEncode(nn.Module):
    """
    out = linear(time_scatter): 1-->time_dims
    out = cos(out)
    """
    def __init__(self, dim):
        super(TimeEncode, self).__init__()
        self.dim = dim
        #self.a = a
        self.w = nn.Linear(1, dim)
        self.reset_parameters()
    
    def reset_parameters(self, ):
        self.w.weight = nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 100, self.dim, dtype=np.float32))).reshape(self.dim, -1))
        self.w.bias = nn.Parameter(torch.zeros(self.dim))

        self.w.weight.requires_grad = False
        self.w.bias.requires_grad = False
    
    @torch.no_grad()
    def forward(self, t):
        output = torch.cos(self.w(t.reshape((-1, 1))))
        return output


class snapshot():
    def __init__(self,time,graph,device):
        self.d = 60
        self.graph = graph
        self.time = time
        self.node_num = graph.num_nodes()
        self.time_enc = TimeEncode(100).to(device)
        self.snapshot_timefeat = self.time_enc(torch.tensor(time).to(torch.float32).to(device))/10
        self.node_edge_mat = 0
        self.edge_list = []
        self.node_edge_mat = 0
        self.time_features = 0
        self.edge_index_list = []
        self.device = device
        self.node_edge_mat = []
        self.edge_list = torch.stack(graph.edges()).t().to(device)
        self.initialize_edge_node_mat()
    # def add_edge(self,graph):
    #     # a = torch.rand(1)
    #     # if a < np.exp(0.00000001*(e[2] - self.time)):
    #     for i in range(graph.edge_index):
    #     self.edge_list.append(torch.tensor(e))
    def initialize_edge_node_mat(self):
        #print(len(self.edge_list))
        self.node_edge_mat = torch.zeros(self.node_num,len(self.edge_list)).to(self.device)
        self.edge_node_mat = torch.zeros(len(self.edge_list),self.node_num).to(self.device)
        
        #self.edge_list = torch.stack(self.edge_list).to(self.device)
        self.time_features = self.time_enc(self.graph.edge_time.to(torch.float32).to(self.device))
        self.time_rate = torch.matmul(self.time_features,self.snapshot_timefeat.t())/10
        self.time_features = 0
        #self.adj_mat = torch.eye(self.node_num).to(self.device)
        graph = self.graph.add_self_loop()
        self.adj_mat = graph.adjacency_matrix()
        
        in_degree = 1/graph.in_degrees()
        degree_matrix = torch.diag(in_degree.float())
        self.inverse_degree_matrix = degree_matrix
        for i,e in enumerate(self.edge_list):
            self.node_edge_mat[int(e[0])][i] = self.time_rate[i]
            self.node_edge_mat[int(e[1])][i] = self.time_rate[i]
            
        
        
        
        # for i in self.node_edge_mat[1]:
        #     if i > 1:
        #         print(i)
        self.node_edge_mat = self.node_edge_mat
    
        #self.node_edge_mat = torch.softmax(self.node_edge_mat,dim=0)
        
    
    #def initialize_
        
        
    
    def set_edge_features(self,feature):
        #edge_index_list = self.edge_list[:,3].to(torch.long)
        #self.edge_feature = feature[edge_index_list].to(self.device)
        node_feature = torch.matmul(self.node_edge_mat,feature.to(self.device))
        #node_feature= torch.cat([node_feature,self.snapshot_timefeat.repeat(node_feature.shape[0],1)],dim=1)
        
        return node_feature.to(torch.device("cpu"))
    
    def set_node_features(self,node_feature):
        #a =  torch.matmul(self.node_edge_mat,self.edge_node_mat)
        self.edge_node_mat = 0
        self.edge_list = []
        self.time_features = 0
        self.node_edge_mat = 0
        a= torch.matmul(self.adj_mat,self.inverse_degree_matrix).to(self.device)
        #self.adj_mat = 0
        return torch.matmul(a,node_feature).to(torch.device("cpu"))

    def empty(self):
        self.node_num = 0
        self.time_enc = 0
        
        self.node_edge_mat = 0
        self.edge_list = []
        self.node_edge_mat = 0
        self.time_features = 0
        self.edge_index_list = []
        self.device = 0
        self.node_edge_mat = []
        self.time_rate = 0

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
      
    
    def forward(self,g,x):

        node_feat = x[g.edge_label_index]
        nodes_1 = node_feat[0]
        nodes_2 = node_feat[1]
        x = torch.cat([nodes_1, nodes_2], dim=1)
        h = torch.sigmoid(self.fc2(self.act(self.fc1(x))))
        return h
    
    # class scalable_tgn_model(nn.Module):
    #     def __init__(self,len_max_seq, embed_dim, d_model, d_inner, n_layers, n_head,
    #             d_k, d_v, dropout,input_dim1,hidden_dim,output_dim,hop) -> None:
    #         super().__init__()
    #         self.encoder = Encoder(len_max_seq, embed_dim, d_model, d_inner, n_layers, n_head,
    #             d_k, d_v, dropout)
    #         self.mlp = Predict_layer(input_dim1,hidden_dim,output_dim,hop)
        
    #     def forward()
def train_scalable_tgn(model, model_transformer,optimizer, device, graph_l, logger, train_n,val_n,test_n,args):
    best_param = {'best_acc': 0, 'best_state': None, 'best_s_dw': None}
    graph_train = graph_l[:train_n]
    graph_val = graph_l[train_n:train_n+1]
    graph_test = graph_l[train_n+1:]
    hop = 3
        #last_timestamp = 0
    last_snapshot = 0
    last_node_feature = 0
    last_two_hop_feature = 0
    last_three_hop_feature = 0
    train_feature_list = []
    val_feature_list = []
    test_feature_list = []
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
        snapshot_i = snapshot(snapshot_timestamp,graph,device=device)
        snapshot_timestamp_list.append(snapshot_i.snapshot_timefeat)
        feature = snapshot_i.set_edge_features(graph.edge_feature).to(device)
        two_hop_feature = snapshot_i.set_node_features(feature).to(device)
        three_hop_feature = snapshot_i.set_node_features(two_hop_feature).to(device)
        # if idx >0:
        #     time_gap = torch.matmul(last_snapshot.snapshot_timefeat,snapshot_i.snapshot_timefeat.t()).to(device)
        #     feature = feature + time_gap*last_node_feature
        #     two_hop_feature = two_hop_feature + time_gap*last_two_hop_feature
        #     three_hop_feature = three_hop_feature+time_gap*last_three_hop_feature
        #last_timestamp= snapshot_timestamp
        train_feature_list.append(feature.cpu())
        train_2hop_feature_list.append(two_hop_feature.cpu())
        train_3hop_feature_list.append(three_hop_feature.cpu())
        last_node_feature= feature
        last_snapshot = snapshot_i
        last_two_hop_feature= two_hop_feature
        last_three_hop_feature = three_hop_feature
    
    for idx, graph in enumerate(graph_val):
        print(idx)
        snapshot_timestamp = graph.edge_time.max()
        snapshot_i = snapshot(snapshot_timestamp,graph,device=device)
        snapshot_timestamp_list.append(snapshot_i.snapshot_timefeat)
        feature = snapshot_i.set_edge_features(graph.edge_feature).to(device)
        two_hop_feature = snapshot_i.set_node_features(feature).to(device)
        three_hop_feature = snapshot_i.set_node_features(two_hop_feature).to(device)
        # time_gap = torch.matmul(last_snapshot.snapshot_timefeat,snapshot_i.snapshot_timefeat.t()).to(device)
        # feature = feature + time_gap*last_node_feature
        val_feature_list.append(feature.cpu())
        # two_hop_feature = two_hop_feature + time_gap*last_two_hop_feature
        # three_hop_feature = three_hop_feature+time_gap*last_three_hop_feature
        val_2hop_feature_list.append(two_hop_feature.cpu())
        val_3hop_feature_list.append(three_hop_feature.cpu())
        #last_timestamp= snapshot_timestamp
        
        last_node_feature= feature
        last_snapshot = snapshot_i
        last_two_hop_feature= two_hop_feature
        last_three_hop_feature = three_hop_feature
    
    for idx, graph in enumerate(graph_test):
        print(idx)
        snapshot_timestamp = graph.edge_time.max()
        snapshot_i = snapshot(snapshot_timestamp,graph,device=device)
        snapshot_timestamp_list.append(snapshot_i.snapshot_timefeat)
        feature = snapshot_i.set_edge_features(graph.edge_feature).to(device)
        two_hop_feature = snapshot_i.set_node_features(feature).to(device)
        three_hop_feature = snapshot_i.set_node_features(feature).to(device)
        # time_gap = torch.matmul(last_snapshot.snapshot_timefeat,snapshot_i.snapshot_timefeat.t()).to(device)
        # feature = feature + time_gap*last_node_feature
        # two_hop_feature = two_hop_feature + time_gap*last_two_hop_feature
        # three_hop_feature = three_hop_feature + time_gap*last_three_hop_feature
        # test_2hop_feature_list.append(two_hop_feature.cpu())
        # test_3hop_feature_list.append(three_hop_feature.cpu())
        #last_timestamp= snapshot_timestamp
        test_feature_list.append(feature.cpu())
        last_node_feature= feature
        last_snapshot = snapshot_i
        last_two_hop_feature= two_hop_feature
        last_three_hop_feature = three_hop_feature
    train_feature_list = torch.stack(train_feature_list)
    val_feature_list = torch.stack(val_feature_list)
    test_feature_list = torch.stack(test_feature_list)
    end_time = time.time()
    print("elapsed time:"+str(end_time-start_time)+"s")
   
    for epoch in range(args.epochs):
        
        print(model_transformer.parameters())
        for idx in range(len(graph_train)-1):

            # for param in model_transformer.parameters():
            #     if param.grad is not None:
            #         print(f"Parameter has gradient:")
            #     else:
            #         print(f"Parameter has no gradient.")

            #graph = graph.to(device)

            input_timefeat_list = torch.stack(snapshot_timestamp_list[:idx+1]).permute(1,0,2)
            input_timefeat_list = input_timefeat_list.repeat(train_feature_list[0].shape[0],1,1)
            hist_node_feature_list = train_feature_list[:idx+1].permute(1,0,2).to(device)
            hist_node_feature_list_with_time = torch.cat([input_timefeat_list,hist_node_feature_list],dim=2)
            output,weight = model_transformer(hist_node_feature_list_with_time)
            #weight = output
            #output = output[:,-1:,:].squeeze(1)
            weight = weight[:,-1,:].unsqueeze(2)
            #weight = weight.unsqueeze(1).unsqueeze(2)
            #print(weight)
            
            # his_feature_list = train_feature_list[:idx+1].to(device)
            weight = weight.repeat(1,1,hist_node_feature_list.shape[2])
            feature = weight*hist_node_feature_list
            feature = torch.sum(feature, dim=1)

            #feature = train_feature_list[idx].to(device)
            #feature = train_feature_list[idx].to(device)
            # two_hop_feature = train_2hop_feature_list[idx].to(device)
            # three_hop_feature = train_3hop_feature_list[idx].to(device)
            #feature = feature + 0.1*two_hop_feature
            # if hop ==2:
            #     feature = torch.cat([feature,two_hop_feature],dim=1)
            # if hop == 3:
            #     feature = torch.cat([feature,two_hop_feature,three_hop_feature],dim=1)
            #pred = model(graph,feature).squeeze(1)
            pred = model(graph_train[idx+1],feature).squeeze(1)
            loss = Link_loss_meta(pred, graph_train[idx+1].edge_label)
            acc, ap, f1, macro_auc, micro_auc = prediction(pred, graph_train[idx+1].edge_label)
            logger.info('meta epoch:{},  loss: {:.5f}, acc: {:.5f}, ap: {:.5f}, f1: {:.5f}, macro_auc: {:.5f}, micro_auc: {:.5f}'.
                            format(epoch,  loss, acc, ap, f1, macro_auc, micro_auc))
            feature =feature.cpu()
           
         
            

            #if losses:
                #losses = losses / count
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        val_acc = 0
        count = 0
        for idx in range(len(graph_val)):
            input_timefeat_list = torch.stack(snapshot_timestamp_list[:idx+len(graph_train)]).permute(1,0,2).repeat(train_feature_list[0].shape[0],1,1)
            hist_node_feature_list = train_feature_list.permute(1,0,2).to(device)
            hist_node_feature_list_time = torch.cat([input_timefeat_list,hist_node_feature_list],dim=2)
            output,weight = model_transformer(hist_node_feature_list_time)
            #weight = output[:,:,0].unsqueeze(2)
            weight = weight[:,-1,:].unsqueeze(2)
            weight = weight.repeat(1,1,hist_node_feature_list.shape[2])
            feature = weight*hist_node_feature_list
            feature = torch.sum(feature, dim=1)
            #val_feature_list = torch.cat(train_feature_list)
            # his_feature_list = train_feature_list.to(device)
            # weight = weight.repeat(1,his_feature_list.shape[1],his_feature_list.shape[2])
            # feature = weight*his_feature_list
            # feature = torch.sum(feature, dim=0)
            #graph = graph.to(device)
            #feature = train_feature_list[-1].to(device)
            # two_hop_feature = train_2hop_feature_list[-1].to(device)
            # three_hop_feature = train_3hop_feature_list[-1].to(device)
            #feature +=  two_hop_feature*0.1
            # if hop ==2:
            #     feature = torch.cat([feature,two_hop_feature],dim=1)
            # if hop == 3:
            #     feature = torch.cat([feature,two_hop_feature,three_hop_feature],dim=1)
            pred = model(graph_val[idx],feature).squeeze(1)
            loss = Link_loss_meta(pred, graph_val[idx].edge_label)
            acc, ap, f1, macro_auc, micro_auc = prediction(pred, graph_val[idx].edge_label)
            logger.info('meta val epoch:{},  loss: {:.5f}, acc: {:.5f}, ap: {:.5f}, f1: {:.5f}, macro_auc: {:.5f}, micro_auc: {:.5f}'.
                            format(epoch,  loss, acc, ap, f1, macro_auc, micro_auc))
            feature =feature.cpu()
            val_acc += acc
            count +=1
           
        
        val_acc = val_acc / count
        if val_acc > best_param['best_acc']:
            best_param = {'best_acc': val_acc, 'best_state': deepcopy(model.state_dict())}
            earl_stop_c = 0
        else:
            earl_stop_c += 1
            if earl_stop_c == 10:
                break
    # testing
    
    model.load_state_dict(best_param['best_state'])
    model.eval()
    avg_acc = 0.0
    for idx in range(len(graph_test)):
        #graph = graph.to(device)
        input_timefeat_list = torch.stack(snapshot_timestamp_list[:idx+len(graph_train)+len(graph_val)]).permute(1,0,2)
        input_timefeat_list = input_timefeat_list.repeat(train_feature_list[0].shape[0],1,1)
        #hist_node_feature_list = torch.cat([train_feature_list,val_feature_list,test_feature_list[:idx]],dim=1)
        # output,weight = model_transformer(hist_node_feature_list)
        # output = output[:,-1:,:].squeeze(1)
        # weight = weight.unsqueeze(1).unsqueeze(2)
        
        
        if idx > 0:
            hist_node_feature_list = test_feature_list[:idx]
            hist_node_feature_list = torch.cat([train_feature_list,val_feature_list,hist_node_feature_list],dim=0).to(device)
            # weight = weight.repeat(1,his_feature_list.shape[1],his_feature_list.shape[2])
            # feature = weight*his_feature_list
            # feature = torch.sum(feature, dim=0)
        else:
            hist_node_feature_list = torch.cat([train_feature_list,val_feature_list],dim=0).to(device)
            
            # weight = weight.repeat(1,his_feature_list.shape[1],his_feature_list.shape[2])
            # feature = weight*his_feature_list
            # feature = torch.sum(feature, dim=0)
        hist_node_feature_list = hist_node_feature_list.permute(1,0,2).to(device)
        hist_node_feature_list_time = torch.cat([input_timefeat_list,hist_node_feature_list],dim=2)
        output,weight = model_transformer(hist_node_feature_list_time)
        weight = weight[:,-1,:].unsqueeze(2)
        #weight = output
        print(weight[0])
        weight = weight.repeat(1,1,hist_node_feature_list.shape[2])
        
        feature = weight*hist_node_feature_list
        feature = torch.sum(feature, dim=1)
        #output = output[:,-1:,:].squeeze(1)
        #feature += 0.1*two_hop_feature
        #two_hop_feature = test_2hop_feature_list[idx].to(device)
        # if hop ==2:
        #     feature = torch.cat([feature,two_hop_feature],dim=1)
        # if hop == 3:
        #     feature = torch.cat([feature,two_hop_feature,three_hop_feature],dim=1)
        pred = model(graph_test[idx],feature).squeeze(1)
        loss = Link_loss_meta(pred, graph_test[idx].edge_label)
        acc, ap, f1, macro_auc, micro_auc = prediction(pred, graph_test[idx].edge_label)
        logger.info('test epoch:{},  loss: {:.5f}, acc: {:.5f}, ap: {:.5f}, f1: {:.5f}, macro_auc: {:.5f}, micro_auc: {:.5f}'.
                        format(epoch,  loss, acc, ap, f1, macro_auc, micro_auc))
        feature =feature.cpu()
        avg_acc += acc
        
    #print("sum"+str(i))
    avg_acc /= (len(graph_test)) 
    logger.info({'avg_acc': avg_acc})
    return avg_acc
    
            
    


    
    

       