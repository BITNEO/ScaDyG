import dgl
import math
import wandb
import torch
import random
import argparse
import numpy as np

from tqdm import tqdm
from model import WinGNN
from test_new import test
from train_new import train
from model.config import cfg
from deepsnap.graph import Graph
from model.Logger import getLogger
from dataset_prep import load, load_r
from model.utils import create_optimizer
from deepsnap.dataset import GraphDataset
from scalable_tgn_weight import train_scalable_tgn,Predict_layer,Encoder
import warnings
import time
warnings.filterwarnings("ignore")





if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='uci-msg', help='Dataset')

    parser.add_argument('--cuda_device', type=int,
                        default=1, help='Cuda device no -1')

    parser.add_argument('--seed', type=int, default=2023, help='split seed')

    parser.add_argument('--repeat', type=int, default=1, help='number of repeat model')

    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train.')

    parser.add_argument('--out_dim', type=int, default=64,
                        help='model output dimension.')

    parser.add_argument('--optimizer', type=str, default='adam',
                        help='optimizer type')

    parser.add_argument('--lr', type=float, default=0.0001,
                        help='initial learning rate.')

    parser.add_argument('--maml_lr', type=float, default=0.008,
                        help='meta learning rate')

    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay (L2 loss on parameters).')

    parser.add_argument('--drop_rate', type=float, default=0.16, help='drop meta loss')

    parser.add_argument('--num_layers', type=int,
                        default=2, help='GNN layer num')

    parser.add_argument('--num_hidden', type=int, default=256,
                        help='number of hidden units of MLP')

    parser.add_argument('--window_num', type=float, default=8,
                        help='windows size')

    parser.add_argument('--dropout', type=float, default=0.1,
                        help='GNN dropout')

    parser.add_argument('--residual', type=bool, default=False,
                        help='skip connection')
    
    parser.add_argument('--model', type=str, default=False,
                        help='skip connection')

    parser.add_argument('--alpha', type=float, default=0.89,
                        help='The weight of adaptive learning rate component accumulation')
    



    args = parser.parse_args()

    logger = getLogger(cfg.log_path)

    # load datasets
    if args.dataset == 'dblp':
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
    elif args.dataset in ["reddit_body", "reddit_title", "as_733",
                          "uci-msg", "bitcoinotc", "bitcoinalpha",
                          'stackoverflow_M','wikipedia','reddit','mooc']:
        graphs, e_feat, e_time, n_feat = load_r(args.dataset)
    else:
        raise ValueError

    n_dim = n_feat[0].shape[1]
    n_node = n_feat[0].shape[0]


    device = torch.device(f'cuda:{args.cuda_device}' if args.cuda_device >= 0 else 'cpu')

    all_mrr_avg = 0.0
    best_mrr = 0.0
    best_model = 0
    all_acc = 0

    for rep in range(args.repeat):

        logger.info('num_layers:{}, num_hidden: {}, lr: {}, maml_lr:{}, window_num:{}, drop_rate:{}, 负样本采样固定'.
                    format(args.num_layers, args.num_hidden, args.lr, args.maml_lr, args.window_num, args.drop_rate))
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
                graph_d.node_feature = torch.Tensor(n_feat_t)
            else:
                graph_d.node_feature = torch.Tensor(n_feat[idx])

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

            dataset = GraphDataset(graph,
                                   task='link_pred',
                                   edge_negative_sampling_ratio=1.0,
                                   minimum_node_per_graph=5)
            edge_labe_index = dataset.graphs[0].edge_label_index
            graph_l[idx].edge_label_index = torch.LongTensor(edge_labe_index)
        #model = Predict_layer(graph.edge_feature.shape[1],64,1,hop=1).to(device)
        model = Predict_layer(graph.edge_feature.shape[1],64,1,hop=1).to(device)
        model_transformer = Encoder(len_max_seq=len(graphs), embed_dim=dataset.graphs[0].edge_feature.shape[1]+100, d_model=100,
                                       d_inner=100, n_layers=2, n_head=8, d_k=64, d_v=64,
                                       dropout=0.1,device=device).to(device)
        model.train()
        model_transformer.train()
        train_n = math.ceil(len(graph_l) * 0.7)
        val_n = train_n+math.ceil(len(graph_l) * 0.1)
        test_n = train_n+1
        parameters = list(model.parameters()) + list(model_transformer.parameters())
        optimizer = create_optimizer(args.optimizer, parameters, args.lr, args.weight_decay)
        start_time = time.time()
        avg_acc = train_scalable_tgn(model, model_transformer,optimizer, device, graph_l, logger,train_n,val_n,test_n,args)
        #model.load_state_dict(best_param['best_state'])
        end_time = time.time()
        print(end_time-start_time)
        all_acc += avg_acc
    all_acc = all_acc/ args.repeat
    print(all_acc)
        
    
        