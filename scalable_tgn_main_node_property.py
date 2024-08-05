import dgl
import math
import wandb
import torch
import random
import argparse
import numpy as np

from tqdm import tqdm
from model import WinGNN

from model.config import cfg
from deepsnap.graph import Graph
from model.Logger import getLogger
from dataset_prep import load, load_r,load_r_without_node
from model.utils import create_optimizer
from deepsnap.dataset import GraphDataset

import warnings
import time
warnings.filterwarnings("ignore")

# trade
# Best trial mrr: 0.6317869592951154
# Best hyperparameters: {'a': 0.034143450512559376, 'lr': 0.00886572355558703, 'n': 26}




def negative_sampling(edges, max_node_id):
    """
    对于每个正边生成一个负边。负边的起点与正边相同，终点随机生成，且确保负边不与现有边重复。
    
    :param edges: 输入的边列表，每个元素是一对 [u, v]
    :param max_node_id: 图中所有节点的最大ID值
    :return: 生成的负边列表，每个元素是一对 [u, v]
    """
    # 转换正边列表为集合形式，便于后续重复性检查
    edges = edges.t().tolist()
    edge_set = set(tuple(edge) for edge in edges)
    negative_edges = []
    max_node_id = max(max(u, v) for u, v in edges)


    for u, _ in edges:
        while True:
            # 随机生成新的终点v
            u_new = random.randint(0, max_node_id)
            v_new = random.randint(0, max_node_id)
            # 检查生成的负边是否与现有边重复
            if (u_new, v_new) not in edge_set:
                negative_edges.append([u_new, v_new])
                break  # 成功生成一个负边后跳出循环
    all_edges = edges + negative_edges
    labels = [1] * len(edges) + [0] * len(negative_edges)

    return all_edges, labels     


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='reddit', help='Dataset')

    parser.add_argument('--cuda_device', type=int,
                        default=0 ,help='Cuda device no -1')

    parser.add_argument('--seed', type=int, default=2023, help='split seed')

    parser.add_argument('--repeat', type=int, default=1, help='number of repeat model')

    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train.')

    parser.add_argument('--out_dim', type=int, default=512,
                        help='model output dimension.')

    parser.add_argument('--optimizer', type=str, default='adam',
                        help='optimizer type')

    parser.add_argument('--lr', type=float, default= 0.0017181244060541123,
                        help='initial learning rate.')
    
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay (L2 loss on parameters).')
    parser.add_argument('--a', type=float, default= 0.08107648597049522,
                        help='The parameter of time encoding')
    parser.add_argument('--weight_share', type=float, default=10,
                        help='The weight of adaptive learning rate component accumulation')
    parser.add_argument('--batch_size', type=int, default=128,
                         help='The weight of adaptive learning rate component accumulation')
    parser.add_argument('--hop', type=float, default=1,
                         help='number of hop used')
    parser.add_argument('--b', type=float, default=12,
                         help='parameter of normalization')
    parser.add_argument('--n', type=int, default=16,
                         help='number of historical length used')
    parser.add_argument('--fusion', type=str, default='t2v',
                         help='ways to fuse temporal and structural information')
    parser.add_argument('--recursive_sum', type=str, default='False',
                         help='ways to recursively sum')
    parser.add_argument('--time_rate', type=int, default=0.05,
                         help='ways to recursively sum')
    parser.add_argument('--base_value', type=int, default=1e-9,
                         help='ways to recursively sum')
    parser.add_argument('--feat_repeat', type=int, default=1,
                         help='if the edge feature == 1 , repreat for n times')
    
    



    args = parser.parse_args()

    logger = getLogger(cfg.log_path)

    # load datasets
    if args.dataset == 'aaaa':
        dataset = args.dataset
        e_feat = np.load('dataset/{0}/ml_{0}.npy'.format(dataset))
        n_feat_ = np.load('dataset/{0}/ml_{0}_node.npy'.format(dataset))
        train_data, train_e_feat, train_n_feat, test_data, test_e_feat, test_n_feat = load("Norandom", len(n_feat_))
        graphs = []
        for tr in train_data:
            graphs.append(tr)
        for te in test_data:
            graphs.append(te)
        n_feat = [n_feat_ for i in range(len(graphs))]
    
    graphs, e_feat, e_time,n_node, n_label = load_r_without_node(args.dataset)
    num_class = list(n_label[list(n_label)[0]].values())[0].shape[0]


    # if e_feat[0].shape[1] == 1:
    #     for i,e in enumerate(e_feat):
            
    #         e=  np.repeat(e , 4, axis=1) 
    #         e_feat[i] = e
        
    
    

    # n_dim = n_feat[0].shape[1]
    # n_node = n_feat[0].shape[0]

    #node_feature = torch.zeros(n_node,n_node)
    if args.dataset == 'trade':
        node_feature =  torch.eye(n_node)
    if args.dataset == 'genre':
        node_feature = torch.rand(n_node,128)
    else:
        node_feature = torch.rand(n_node,128)
    

    device = torch.device(f'cuda:{args.cuda_device}' if args.cuda_device >= 0 else 'cpu')

    all_mrr_avg = 0.0
    best_mrr = 0.0
    best_model = 0
    all_ndcg = 0
    all_mse = 0
    all_auc = 0
    all_start_time = time.time()
    for rep in range(args.repeat):

        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)

        graph_l = []
        
        # Data set processing
        for idx, graph in tqdm(enumerate(graphs)):
            graph_d = dgl.from_scipy(graph)

            graph_d.edge_feature = torch.Tensor(e_feat[idx]).t()
            graph_d.edge_time = torch.Tensor(e_time[idx])

            # if n_feat[idx].shape[0] != n_node or n_feat[idx].shape[1] != n_dim:
            #     n_feat_t = graph_l[idx - 1].node_feature
            #     graph_d.node_feature = torch.Tensor(n_feat_t)
            # else:
            graph_d.node_feature = torch.Tensor(node_feature)

            #graph_d = dgl.remove_self_loop(graph_d)
            #graph_d = dgl.add_self_loop(graph_d)

            edges = graph_d.edges()
            row = edges[0].numpy()
            col = edges[1].numpy()
            # Negative sample sampling 1:1
            n_e = graph_d.num_edges() 
            # Edge label
            y_pos = np.ones(shape=(n_e,))
            y_neg = np.zeros(shape=(n_e,))
            y = list(y_pos) + list(y_neg)

            edge_label_index = list()
            edge_label_index.append(row.tolist()[:n_e])
            edge_label_index.append(col.tolist()[:n_e])

            graph_d.edge_label = torch.Tensor(y)
            graph_d.edge_label_index = torch.LongTensor(edge_label_index)

            graph_l.append(graph_d)
        # Negative sample sampling 1:1
        # for idx, graph in tqdm(enumerate(graphs)):
        #     graph = Graph(
        #         node_feature=graph_l[idx].node_feature,
        #         edge_feature=graph_l[idx].edge_feature,
        #         edge_index=graph_l[idx].edge_label_index,
        #         edge_time=graph_l[idx].edge_time,
        #         directed=True
        #     )
        #     edge_labe_index = graph_l[idx].edge_label_index
        #     dataset = GraphDataset(graph,
        #                            task='link_pred',
        #                            edge_negative_sampling_ratio=1.0,
        #                            minimum_node_per_graph=5)
            
            # edge_label_index,edge_label = negative_sampling(edge_labe_index,graph.num_nodes-1)
            # graph_l[idx].edge_label_index = torch.LongTensor(edge_label_index).t()
            # edge_labe_index = dataset.graphs[0].edge_label_index
            # graph_l[idx].edge_label_index = torch.LongTensor(edge_labe_index)
        #model = Predict_layer(graph.edge_feature.shape[1],64,1,hop=1).to(device)
        if args.fusion == 'v2t':
            from scalable_tgn_affine_v2t_chunk import train_scalable_tgn,NodePredictor,Encoder
        else:
            from scalable_tgn_node_pred import train_scalable_tgn,NodePredictor,Encoder
        if args.dataset in  ['reddit_title','USLegis','UNovte','SocialEvo','trade','genre','token','reddit']:
            #model = NodePredictor(graph_l[idx].edge_feature.shape[1]+graph_l[idx].node_feature.shape[1],128,num_class).to(device)
            model = NodePredictor(graph_l[idx].node_feature.shape[1],128,num_class).to(device)
            model_transformer = Encoder(embed_dim_1=graph_l[idx].node_feature.shape[1], embed_dim_2=graph_l[idx].edge_feature.shape[1]+graph_l[idx].node_feature.shape[1],d_model=64,
                                            d_inner=graph_l[idx].edge_feature.shape[1]+graph_l[idx].node_feature.shape[1], n_layers=1, n_head=8, d_k=64, d_v=64,
                                            dropout=0.1,device=device).to(device)
            # model_transformer = Encoder(embed_dim_1=100, embed_dim_2=dataset.graphs[0].edge_feature.shape[1]+graph.node_feature.shape[1],d_model=64,
            #                                 d_inner=1, n_layers=1, n_head=8, d_k=64, d_v=64,
            #                                 dropout=0.1,device=device).to(device)
        else:
            model = NodePredictor(graph_l[idx].node_feature.shape[1],64,num_class).to(device)
            model_transformer = Encoder( embed_dim_1=graph_l[idx].edge_feature.shape[1], embed_dim_2=graph_l[idx].edge_feature.shape[1],d_model=64,
                                            d_inner=graph_l[idx].edge_feature.shape[1], n_layers=1, n_head=8, d_k=64, d_v=64,
                                            dropout=0.1,device=device).to(device)
            # model_transformer = Encoder(embed_dim_1=100, embed_dim_2=dataset.graphs[0].edge_feature.shape[1],d_model=64,
            #                                 d_inner=1, n_layers=1, n_head=8, d_k=64, d_v=64,
            #                                 dropout=0.1,device=device).to(device)

        # model_transformer_1 = Encoder(len_max_seq=len(graphs), embed_dim=dataset.graphs[0].edge_feature.shape[1], d_model=100,  
        #                                 d_inner=1, n_layers=2, n_head=8, d_k=64, d_v=64,
        #                                 dropout=0.1,device=device).to(device)
        # model_transformer_2= Encoder(len_max_seq=len(graphs), embed_dim=2, d_model=100,
        #                                 d_inner=1, n_layers=2, n_head=8, d_k=64, d_v=64,
        #                                 dropout=0.1,device=device).to(device)
        model.train()
        model_transformer.train()

        total_params_trm = sum(p.numel() for p in model_transformer.parameters())
        total_params_linear = sum(p.numel() for p in model.parameters())


        print(f"Total number of transformer parameters is: {total_params_trm}")
        print(f"Total number of predit_layer parameters is: {total_params_linear}")
        print(total_params_trm+total_params_linear)
        # model_transformer_1.train()
        # model_transformer_2.train()
        train_n = math.ceil(len(graph_l) * 0.7)
        val_n = train_n+math.ceil(len(graph_l) * 0.15)
        test_n = train_n+1
        parameters = list(model.parameters()) + list(model_transformer.parameters())
        optimizer = create_optimizer(args.optimizer, parameters, args.lr, args.weight_decay)
        start_time = time.time()
        
        avg_ndcg,avg_mse = train_scalable_tgn(model, model_transformer,optimizer, device, graph_l,n_label, logger,train_n,val_n,test_n,args)
        #model.load_state_dict(best_param['best_state'])
        end_time = time.time()
        print(end_time-start_time)
        all_ndcg+=avg_ndcg
        all_mse+=avg_mse
        
    
    all_ndcg = all_ndcg/ args.repeat
    all_mse = all_mse/ args.repeat
    all_end_time = (time.time()-all_start_time)/args.repeat
    logger.info(f"All ndcg: {all_mse}")
    logger.info(f"All mse: {all_auc}")
    logger.info(f"All time: {all_end_time}")
        
    
        