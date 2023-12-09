import argparse
import math
import os.path



import numpy as np
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score
from torch.nn import BCEWithLogitsLoss, Conv1d, MaxPool1d, ModuleList

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MLP, GCNConv, SortAggregation
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import to_scipy_sparse_matrix, k_hop_subgraph

from torch_geometric.datasets import RelLinkPredDataset

from torch_geometric.data import Dataset
from torch_geometric.utils.num_nodes import maybe_num_nodes
from tqdm import tqdm
import time
import torch
from line_profiler_pycharm import profile

from torch_geometric.utils import add_self_loops

def train(model,train_loader,train_dataset,optimizer,criterion,device):
    model.train()

    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out.view(-1), data.y.to(torch.float))
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * data.num_graphs

    return total_loss / len(train_dataset)

@torch.no_grad()
def test(model,loader,device):
    model.eval()

    y_pred, y_true = [], []
    for data in loader:
        data = data.to(device)
        logits = model(data.x, data.edge_index, data.batch)
        y_pred.append(logits.view(-1).cpu())
        y_true.append(data.y.view(-1).cpu().to(torch.float))

    return roc_auc_score(torch.cat(y_true), torch.cat(y_pred))

def load_args():
    parser = argparse.ArgumentParser(description='Parser For Arguments',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-name', default='testrun', help='Set run name for saving/restoring models')
    parser.add_argument('-data', dest='dataset', default='FB15k-237', help='Dataset to use, default: FB15k-237')
    parser.add_argument('-model', dest='model', default='compgcn', help='Model Name')
    parser.add_argument('-score_func', dest='score_func', default='conve', help='Score Function for Link prediction')
    parser.add_argument('-opn', dest='opn', default='corr', help='Composition Operation to be used in CompGCN')

    parser.add_argument('-batch', dest='batch_size', default=128, type=int, help='Batch size')
    parser.add_argument('-gamma', type=float, default=40.0, help='Margin')
    parser.add_argument('-gpu', type=str, default='0', help='Set GPU Ids : Eg: For CPU = -1, For Single GPU = 0')
    parser.add_argument('-epoch', dest='max_epochs', type=int, default=500, help='Number of epochs')
    parser.add_argument('-l2', type=float, default=0.0, help='L2 Regularization for Optimizer')
    parser.add_argument('-lr', type=float, default=0.001, help='Starting Learning Rate')
    parser.add_argument('-lbl_smooth', dest='lbl_smooth', type=float, default=0.1, help='Label Smoothing')
    parser.add_argument('-num_workers', type=int, default=10, help='Number of processes to construct batches')
    parser.add_argument('-seed', dest='seed', default=41504, type=int, help='Seed for randomization')

    parser.add_argument('-restore', dest='restore', action='store_true', help='Restore from the previously saved model')
    parser.add_argument('-bias', dest='bias', action='store_true', help='Whether to use bias in the model')

    parser.add_argument('-num_bases', dest='num_bases', default=-1, type=int,
                        help='Number of basis relation vectors to use')
    parser.add_argument('-init_dim', dest='init_dim', default=100, type=int,
                        help='Initial dimension size for entities and relations')
    parser.add_argument('-gcn_dim', dest='gcn_dim', default=200, type=int, help='Number of hidden units in GCN')
    parser.add_argument('-embed_dim', dest='embed_dim', default=None, type=int,
                        help='Embedding dimension to give as input to score function')
    parser.add_argument('-gcn_layer', dest='gcn_layer', default=1, type=int, help='Number of GCN Layers to use')
    parser.add_argument('-gcn_drop', dest='dropout', default=0.1, type=float, help='Dropout to use in GCN Layer')
    parser.add_argument('-hid_drop', dest='hid_drop', default=0.3, type=float, help='Dropout after GCN')

    # ConvE specific hyperparameters
    parser.add_argument('-hid_drop2', dest='hid_drop2', default=0.3, type=float, help='ConvE: Hidden dropout')
    parser.add_argument('-feat_drop', dest='feat_drop', default=0.3, type=float, help='ConvE: Feature Dropout')
    parser.add_argument('-k_w', dest='k_w', default=10, type=int, help='ConvE: k_w')
    parser.add_argument('-k_h', dest='k_h', default=20, type=int, help='ConvE: k_h')
    parser.add_argument('-num_filt', dest='num_filt', default=200, type=int,
                        help='ConvE: Number of filters in convolution')
    parser.add_argument('-ker_sz', dest='ker_sz', default=7, type=int, help='ConvE: Kernel size to use')

    parser.add_argument('-logdir', dest='log_dir', default='./log/', help='Log directory')
    parser.add_argument('-config', dest='config_dir', default='./config/', help='Config directory')
    args = parser.parse_args()

    return args

@profile
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_path = osp.join(osp.dirname(osp.realpath(__file__)), '', 'data', 'RLPD')
    data = RelLinkPredDataset(data_path, 'FB15k-237')[0]

    transform = RandomLinkSplit(num_val=0.2, num_test=0.3,
                                is_undirected=True, split_labels=True)
    non_split_data = Data(edge_index=data.edge_index, edge_type=data.edge_type, num_nodes=data.num_nodes)
    train_data, val_data, test_data = transform(non_split_data)

    train_data.pos_edge_label_type = train_data.edge_type[:train_data.pos_edge_label_index.shape[1]]
    train_data.neg_edge_label_type = train_data.edge_type[train_data.pos_edge_label_index.shape[1]:]
    val_data.pos_edge_label_type = val_data.edge_type[:val_data.pos_edge_label_index.shape[1]]
    val_data.neg_edge_label_type = val_data.edge_type[val_data.pos_edge_label_index.shape[1]:]
    test_data.pos_edge_label_type = test_data.edge_type[:test_data.pos_edge_label_index.shape[1]]
    test_data.neg_edge_label_type = test_data.edge_type[test_data.pos_edge_label_index.shape[1]:]

    optimize_mem = True if torch.cuda.is_available() else False
    non_blocking = False
    if optimize_mem:
        train_data.pin_memory()
        test_data.pin_memory()
        val_data.pin_memory()
        non_blocking = True

    train_data = train_data.to(device, non_blocking=non_blocking)
    val_data = val_data.to(device, non_blocking=non_blocking)
    test_data = test_data.to(device, non_blocking=non_blocking)

    train_dataset = RLPDDataset(train_data, num_hops=2, split='train', device=device)
    val_dataset = RLPDDataset(val_data, num_hops=2, split='val', device=device)
    test_dataset = RLPDDataset(test_data, num_hops=2, split='test', device=device)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)


    # model = DGCNN(hidden_channels=32, num_layers=3,train_dataset=train_dataset).to(device)
    model = None
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.0001)
    criterion = BCEWithLogitsLoss()

    best_val_auc = test_auc = 0
    for epoch in range(1, 51):
        loss = train(optimizer=optimizer,criterion=criterion,train_loader=train_loader,train_dataset=train_dataset,device=device)
        val_auc = test(model,val_loader, device=device)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            test_auc = test(test_loader)
        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Val: {val_auc:.4f}, '
              f'Test: {test_auc:.4f}')


if __name__ == '__main__':
    main()