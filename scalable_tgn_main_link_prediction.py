import dgl
import math
import torch
import random
import argparse
import numpy as np

from tqdm import tqdm


from model.config import cfg
from deepsnap.graph import Graph
from model.Logger import getLogger
from dataset_prep import load, load_r
from model.utils import create_optimizer
from deepsnap.dataset import GraphDataset

import warnings
import time
warnings.filterwarnings("ignore")



# best parameters for v2t,uci_msg: {'a': 169, 'weight_share': 19, 'lr': 0.00011654048009358208}. Best is trial 81 with value: 0.885202496530548
# def negative_sampling(edges, max_node_id):
#     """
#     对于每个正边生成一个负边。负边的起点与正边相同，终点随机生成，且确保负边不与现有边重复。
    
#     :param edges: 输入的边列表，每个元素是一对 [u, v]
#     :param max_node_id: 图中所有节点的最大ID值
#     :return: 生成的负边列表，每个元素是一对 [u, v]
#     """
#     # 转换正边列表为集合形式，便于后续重复性检查
#     edges = edges.t().tolist()
#     edge_set = set(tuple(edge) for edge in edges)
#     negative_edges = []
#     max_node_id = max(max(u, v) for u, v in edges)


#     for u, _ in edges:
#         while True:
#             # 随机生成新的终点v
#             v_new = random.randint(0, max_node_id)
#             # 检查生成的负边是否与现有边重复
#             if (u, v_new) not in edge_set:
#                 negative_edges.append([u, v_new])
#                 break  # 成功生成一个负边后跳出循环
#     all_edges = edges + negative_edges
#     labels = [1] * len(edges) + [0] * len(negative_edges)

#     return all_edges, labels        

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
# 
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mooc', help='Dataset')

    parser.add_argument('--cuda_device', type=int,
                        default=1 ,help='Cuda device no -1')

    parser.add_argument('--seed', type=int, default=2023, help='split seed')

    parser.add_argument('--repeat', type=int, default=1, help='number of repeat model')

    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train.')

    parser.add_argument('--out_dim', type=int, default=512,
                        help='model output dimension.')

    parser.add_argument('--optimizer', type=str, default='adam',
                        help='optimizer type')

    parser.add_argument('--lr', type=float, default= 0.03371303354056322,
                        help='initial learning rate.')
    
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay (L2 loss on parameters).')
    parser.add_argument('--a', type=float, default= 9.772233221749005e-07,
                        help='The parameter of time encoding')
    parser.add_argument('--weight_share', type=float, default=10,
                        help='The weight of adaptive learning rate component accumulation')
    parser.add_argument('--batch_size', type=int, default=128,
                         help='The weight of adaptive learning rate component accumulation')
    parser.add_argument('--hop', type=float, default=1,
                         help='number of hop used')
    parser.add_argument('--b', type=float, default=12,
                         help='parameter of normalization')
    parser.add_argument('--n', type=int, default=39,
                         help='number of historical length used')
    #n = 24
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

    if args.dataset == 'bitcoinalpha':
        args.a = 9.772233221749005e-07
        args.lr=0.03371303354056322
        args.n=39
    if args.dataset == 'lastfm':
        args.a = 2.2661581547411702e-07
        args.lr= 0.002758079478978358
        args.n= 32
    if args.dataset == 'UNvote':
        args.a = 2.2661581547411702e-01
        args.lr= 0.002758079478978358
        args.n= 10
    
    if args.dataset == 'Reddit_title':
        args.a =9.063130587806096e-06
        args.lr= 0.0005502332748490608
        args.n= 36
    
    if args.dataset == 'enron':
        args.a =8.801956776817994e-07
        args.lr= 0.024951393100015522
        args.n= 24
    if args.dataset == 'uci':
        args.a =9.796232886499929e-07
        args.lr= 0.004633535643731563
        args.n= 10
    # if args.dataset == 'Flights':
    #     args.a =0.09297283115579853
    #     args.lr= 0.00015254698823245332
    #     args.n= 16
    if args.dataset == 'mooc':
        args.a = 5.6790098270506866e-08
        args.lr= 0.0034562317384750716
        args.n= 9
    
    
    graphs, e_feat, e_time, n_feat = load_r(args.dataset)

    if e_feat[0].shape[1] == 1:
        for i,e in enumerate(e_feat):
            
            e=  np.repeat(e , 2, axis=1) 
            e_feat[i] = e
        
    
    

    n_dim = n_feat[0].shape[1]
    n_node = n_feat[0].shape[0]

    if args.dataset == 'enron':
        node_feature = torch.eye(n_node)
    if args.dataset == 'Flights':
        node_feature = torch.rand(n_node,128)
    else:
        node_feature = torch.rand(n_node,128)
    device = torch.device(f'cuda:{args.cuda_device}' if args.cuda_device >= 0 else 'cpu')

    all_mrr_avg = 0.0
    best_mrr = 0.0
    best_model = 0
    all_acc = 0
    all_ap = 0
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

            graph_d.edge_feature = torch.Tensor(e_feat[idx])
            graph_d.edge_time = torch.Tensor(e_time[idx])

            if n_feat[idx].shape[0] != n_node or n_feat[idx].shape[1] != n_dim:
                n_feat_t = graph_l[idx - 1].node_feature
                graph_d.node_feature = torch.Tensor(node_feature)
            else:
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
        for idx, graph in tqdm(enumerate(graphs)):
            graph = Graph(
                node_feature=graph_l[idx].node_feature,
                edge_feature=graph_l[idx].edge_feature,
                edge_index=graph_l[idx].edge_label_index,
                edge_time=graph_l[idx].edge_time,
                directed=True
            )
            edge_labe_index = graph_l[idx].edge_label_index
            dataset = GraphDataset(graph,
                                   task='link_pred',
                                   edge_negative_sampling_ratio=1.0,
                                   minimum_node_per_graph=5)
            
            edge_label_index,edge_label = negative_sampling(edge_labe_index,graph.num_nodes-1)
            graph_l[idx].edge_label_index = torch.LongTensor(edge_label_index).t()
            # edge_labe_index = dataset.graphs[0].edge_label_index
            # graph_l[idx].edge_label_index = torch.LongTensor(edge_labe_index)
        #model = Predict_layer(graph.edge_feature.shape[1],64,1,hop=1).to(device)
        if args.fusion == 'v2t':
            from scalable_tgn_affine_v2t_chunk import train_scalable_tgn,Predict_layer,Encoder
        else:
            from scalable_tgn_link_prediction import train_scalable_tgn,Predict_layer,Encoder
        if args.dataset in  ['reddit_title','USLegis','UNovte','SocialEvo','Flights','enron']:
            model = Predict_layer(graph.edge_feature.shape[1]+graph.node_feature.shape[1],128,1,hop=args.hop).to(device)
            model_transformer = Encoder(embed_dim_1=dataset.graphs[0].edge_feature.shape[1]+dataset.graphs[0].node_feature.shape[1], embed_dim_2=dataset.graphs[0].edge_feature.shape[1]+graph.node_feature.shape[1],d_model=64,
                                            d_inner=graph.edge_feature.shape[1]+graph.node_feature.shape[1], n_layers=1, n_head=8, d_k=64, d_v=64,
                                            dropout=0.1,device=device).to(device)
            # model_transformer = Encoder(embed_dim_1=100, embed_dim_2=dataset.graphs[0].edge_feature.shape[1]+graph.node_feature.shape[1],d_model=64,
            #                                 d_inner=1, n_layers=1, n_head=8, d_k=64, d_v=64,
            #                                 dropout=0.1,device=device).to(device)
        else:
            model = Predict_layer(graph.edge_feature.shape[1],64,1,hop=args.hop).to(device)
            model_transformer = Encoder( embed_dim_1=dataset.graphs[0].edge_feature.shape[1], embed_dim_2=dataset.graphs[0].edge_feature.shape[1],d_model=64,
                                            d_inner=graph.edge_feature.shape[1], n_layers=1, n_head=8, d_k=64, d_v=64,
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
        
        avg_ap,auc,acc = train_scalable_tgn(model, model_transformer,optimizer, device, graph_l, logger,train_n,val_n,test_n,args)
        #model.load_state_dict(best_param['best_state'])
        end_time = time.time()
        print(end_time-start_time)
        all_ap += avg_ap
        all_acc += acc
        all_auc += auc
    all_acc = all_acc/ args.repeat
    all_auc = all_auc/ args.repeat
    all_ap = all_ap/ args.repeat
    all_end_time = (time.time()-all_start_time)/args.repeat
    logger.info(f"All accuracy: {all_acc}")
    logger.info(f"All AUC: {all_auc}")
    logger.info(f"All AP: {all_ap}")
    logger.info(f"All time: {all_end_time}")
        
    
        