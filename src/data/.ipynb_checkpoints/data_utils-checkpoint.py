import os
import re
import math
import random
import numpy as np
import torch
from pathlib import Path
import os.path as osp
from typing import Union, List, Tuple
from tqdm import tqdm
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.data import Dataset
from torch_geometric.datasets import Planetoid, Amazon, Coauthor, CoraFull, WebKB, Actor, WikipediaNetwork
from torch_geometric.io.planetoid import index_to_mask
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import to_undirected
from src.data.split import get_idx_split

from deeprobust.graph.data import Dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Run at console -> python -c 'from src.data.data_utils import *; split_data("Cora", 5, 3, 85)'
def split_data(
        name: str, 
        samples_in_one_fold: int, 
        k_fold: int, 
        test_samples_per_class: int):
    """
    name: str, the name of the dataset
    samples_in_one_fold: int, sample x% of each class to one fold   
    k_fold: int, k-fold cross validation. One fold is used as validation the rest portions are used as training
    test_samples_per_class: int, sample x% of each class for test set
    """
    print(name)
    assert name in ['Cora','Citeseer', 'Pubmed', 'Computers', 'Photo', 'CS', 'Physics', 'CoraFull', 'Cornell', 'Texas', 'Wisconsin', 'Actor', 'chameleon', 'squirrel']
    if name in ['Cora','Citeseer', 'Pubmed']:
        dataset = Planetoid(root='./data/', name=name, split='random')
    elif name in ['Computers', 'Photo']:
        dataset = Amazon(root='./data/', name=name)
    elif name in ['CS', 'Physics']:
        dataset = Coauthor(root='./data/', name=name)
    elif name == 'CoraFull':
        dataset = CoraFull(root='./data/')
    elif name in ['Cornell', 'Texas', 'Wisconsin']:
        dataset = WebKB(root='./data/', name=name)
    elif name == 'Actor':
        dataset = Actor(root='./data/Actor')
    elif name in ['chameleon', 'squirrel']:
        dataset = WikipediaNetwork(root='./data/', name=name)

    split_type = str(samples_in_one_fold)+"_"+str(k_fold)+'f_'+str(test_samples_per_class)       
    raw_dir = Path(os.path.join('data','split', str(name), split_type))
    raw_dir.mkdir(parents=True, exist_ok=True)

    # For each configuration we split the data five times
    data = dataset._data
    for i in range(5):
        assert int(samples_in_one_fold)*int(k_fold)+int(test_samples_per_class) <= 100, "Invalid fraction" 
        k_fold_indices, test_indices = get_idx_split(data,
                    samples_per_class_in_one_fold=samples_in_one_fold/100.,
                    k_fold=k_fold,
                    test_samples_per_class=test_samples_per_class/100.)
        split_file = f'{name.lower()}_split_{i}.npz'
        print(f"sample/fold/test: {len(k_fold_indices[0])}/{len(k_fold_indices)}/{len(test_indices)}")
        # print(k_fold_indices[0].shape)
        # print(k_fold_indices[1].shape)
        # print(k_fold_indices[2].shape)
        # print(k_fold_indices[3].shape)
        # # print(k_fold_indices)
        # # print(type(k_fold_indices))
        # print(test_indices.shape)
        # print(type(test_indices))
        # np.savez(raw_dir/split_file, k_fold_indices=k_fold_indices, test_indices=test_indices)
        np.savez(raw_dir/split_file, **{f'fold_{i}': arr for i, arr in enumerate(k_fold_indices)}, test_indices=test_indices)

def load_data(name: str, split_type: str, split: int, fold: int) -> Dataset:
    """
    name: str, the name of the dataset
    split_type: str, format {sample per fold ratio}_{k fold}_{test ratio}. For example, 5_3f_85
    split: int, index of the split. In total five splits were generated for each dataset. 
    fold: int, index of the fold to be used as validation set. The rest k-1 folds will be used as training set.
    """
    transform = NormalizeFeatures()
    if name in ['Cora','Citeseer', 'Pubmed']:
        dataset = Planetoid(root='./data/', name=name, transform=transform)
        load_split_from_numpy_files(dataset, name, split_type, split, fold)
    elif name in ['Computers', 'Photo']:
        dataset = Amazon(root='./data/', name=name, transform=transform)
        load_split_from_numpy_files(dataset, name, split_type, split, fold)
    elif name in ['CS', 'Physics']:
        dataset = Coauthor(root='./data/', name=name, transform=transform)
        load_split_from_numpy_files(dataset, name, split_type, split, fold)
    elif name == 'CoraFull':
        dataset = CoraFull(root='./data/', transform=transform)
        load_split_from_numpy_files(dataset, name, split_type, split, fold)
    elif name in ['Cornell', 'Texas', 'Wisconsin']:
        dataset = WebKB(root='./data/', name=name)
        load_split_from_numpy_files(dataset, name, split_type, split, fold)
    elif name == 'Actor':
        dataset = Actor(root='./data/Actor')
        load_split_from_numpy_files(dataset, name, split_type, split, fold)
    elif name in ['chameleon', 'squirrel']:
        dataset = WikipediaNetwork(root='./data/', name=name)
        load_split_from_numpy_files(dataset, name, split_type, split, fold)
    
    return dataset

def load_split_from_numpy_files(dataset, name, split_type, split, fold):
    """
    load train/val/test from saved k-fold split files
    """
    raw_dir = Path(os.path.join('data','split', str(name), split_type))
    # assert raw_dir.is_dir(), "Split type does not exist."
    samples_in_one_fold = int(split_type.split("_")[0])
    k_fold = int(split_type.split("_")[1].replace("f",""))
    test_samples_per_class  = int(split_type.split("_")[2])
    if not os.path.exists(raw_dir):
        split_data(name, samples_in_one_fold, k_fold, test_samples_per_class)
    split_file = f'{name.lower()}_split_{split}.npz'
    masks = np.load(raw_dir / split_file, allow_pickle=True)
    # val_indices = masks['k_fold_indices'][fold]
    # train_indices = np.concatenate(np.delete(masks['k_fold_indices'], fold, axis=0))
    val_fold_name = f'fold_{fold}'
    val_indices = masks[val_fold_name]
    arrays_to_concat = [arr for key, arr in masks.items() if key not in [val_fold_name, 'test_indices']]
    train_indices = np.concatenate(arrays_to_concat)
    test_indices = masks['test_indices']
    # dataset.data.train_mask = index_to_mask(train_indices, dataset.data.num_nodes)
    # dataset.data.val_mask = index_to_mask(val_indices, dataset.data.num_nodes)
    # dataset.data.test_mask = index_to_mask(test_indices, dataset.data.num_nodes)
    dataset._data.train_mask = index_to_mask(torch.tensor(train_indices).flatten(), dataset._data.num_nodes)  #xxt
    dataset._data.val_mask = index_to_mask(torch.tensor(val_indices).flatten(), dataset._data.num_nodes)   #xxt
    dataset._data.test_mask = index_to_mask(torch.tensor(test_indices).flatten(), dataset._data.num_nodes)  #xxt

# Run at console -> python -c 'from src.data.data_utils import *; generate_node_to_nearest_training("Cora", "5_3f_85")'
def generate_node_to_nearest_training(name: str, split_type: str, bfs_depth = 10):
    max_split = 5
    max_fold = int(split_type.split("_")[1].replace("f",""))
    for split in tqdm(range(max_split)):
        raw_dir = Path(os.path.join('data','dist_to_train', str(name), split_type))
        for fold in tqdm(range(max_fold)):
            dataset = load_data(name=name, split_type=split_type, split=split, fold=fold)
            data = dataset._data
            dist_to_train = torch.ones(data.num_nodes) * bfs_depth
            dist_to_train = shortest_path_length(data.edge_index, data.train_mask, bfs_depth)
            raw_split_dir = raw_dir / f'split_{split}'
            raw_split_dir.mkdir(parents=True, exist_ok=True)
            split_file = f'{name.lower()}_dist_to_train_f{fold}.npy'
            np.save(raw_split_dir/split_file, dist_to_train)

def load_node_to_nearest_training(name: str, split_type: str, split: int, fold: int):
    split_file = os.path.join(
        'data', 'dist_to_train', str(name), split_type, f'split_{split}',
        f'{name.lower()}_dist_to_train_f{fold}.npy')
    if not os.path.isfile(split_file):
        generate_node_to_nearest_training(name, split_type)
    return torch.from_numpy(np.load(split_file))

def shortest_path_length(edge_index, mask, max_hop):
    """
    Return the shortest path length to the mask for every node
    """
    dist_to_train = torch.ones_like(mask, dtype=torch.long, device=mask.device) * torch.iinfo(torch.long).max
    seen_mask = torch.clone(mask)
    for hop in range(max_hop):
        current_hop = torch.nonzero(mask)
        dist_to_train[mask] = hop
        next_hop = torch.zeros_like(mask, dtype=torch.bool, device=mask.device)
        for node in current_hop:
            node_mask = edge_index[0,:]==node
            nbrs = edge_index[1,node_mask]
            next_hop[nbrs] = True
        hop += 1
        # mask for the next hop shouldn't be seen before
        mask = torch.logical_and(next_hop, ~seen_mask)
        seen_mask[next_hop] = True
    return dist_to_train        

def get_train_hop_hist(
        edge_index: np.ndarray, train_index: np.ndarray, nodes: int,
        max_hop: int
) -> np.ndarray:
    train_hop_count = np.zeros([nodes, max_hop + 1], dtype=np.int32)
    for t in train_index:
        hops = np.full(nodes, fill_value=max_hop, dtype=np.int32)
        current_nodes = {t}
        seen_nodes = set()
        for h in range(max_hop):
            if not current_nodes:
                break
            current_idx = np.asarray(list(current_nodes))
            hops[current_idx] = h
            seen_nodes |= current_nodes
            next_nodes = set()
            for n in current_nodes:
                next_nodes |= set(
                    edge_index[1, edge_index[0, :] == n].tolist()
                ) - seen_nodes
            current_nodes = next_nodes
        train_hop_count[np.arange(nodes), hops] += 1
    return train_hop_count

def load_train_hop_hist(
        name: str, split_type: str, split: int, fold: int, max_hop: int
) -> Tensor:
    dataset = load_data(name, split_type, split, fold)
    cache_dir = os.path.join(
        'data', 'train_hop_dist', str(name), split_type)
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    cache_name = os.path.join(cache_dir, f's{split}_f{fold}_h{max_hop}.npy')
    if os.path.isfile(cache_name):
        print(f'loading train_hop_dist from {cache_name}')
        return torch.from_numpy(np.load(cache_name)).to(torch.get_default_dtype())
    else:
        print(f'computing train_hop_dist ...')
        data = dataset._data
        nodes = data.num_nodes
        train_index = np.arange(nodes)[data.train_mask.cpu().numpy()]
        train_hop_dist = get_train_hop_hist(
            data.edge_index.cpu().numpy(), train_index, nodes, max_hop)
        print(f'saving computed train_hop_dist to {cache_name}')
        np.save(cache_name, train_hop_dist)
        return torch.from_numpy(train_hop_dist).to(torch.get_default_dtype())

class CustomDataset(Dataset):
    def __init__(self, root, name, setting='gcn', seed=None, require_mask=False):
        '''
        Adopted from https://github.com/GemsLab/H2GCN/blob/master/npz-datasets/dataset.py
        '''
        self.name = name.lower()
        self.setting = setting.lower()

        self.seed = seed
        self.url = None
        self.root = osp.expanduser(osp.normpath(root))
        self.data_folder = osp.join(root, self.name)
        self.data_filename = self.data_folder + '.npz'
        # Make sure dataset file exists
        assert osp.exists(self.data_filename), f"{self.data_filename} does not exist!" 
        self.require_mask = require_mask

        self.require_lcc = True if setting == 'nettack' else False
        self.adj, self.features, self.labels = self.load_data()
        self.idx_train, self.idx_val, self.idx_test = self.get_train_val_test()
        if self.require_mask:
            self.get_mask()
    
    def get_adj(self):
        adj, features, labels = self.load_npz(self.data_filename)
        adj = adj + adj.T
        adj = adj.tolil()
        adj[adj > 1] = 1

        if self.require_lcc:
            lcc = self.largest_connected_components(adj)

            # adj = adj[lcc][:, lcc]
            adj_row = adj[lcc]
            adj_csc = adj_row.tocsc()
            adj_col = adj_csc[:, lcc]
            adj = adj_col.tolil()

            features = features[lcc]
            labels = labels[lcc]
            assert adj.sum(0).A1.min() > 0, "Graph contains singleton nodes"

        # whether to set diag=0?
        adj.setdiag(0)
        adj = adj.astype("float32").tocsr()
        adj.eliminate_zeros()

        assert np.abs(adj - adj.T).sum() == 0, "Input graph is not symmetric"
        assert adj.max() == 1 and len(np.unique(adj[adj.nonzero()].A1)) == 1, "Graph must be unweighted"

        return adj, features, labels
    
    def get_train_val_test(self):
        if self.setting == "exist":
            with np.load(self.data_filename) as loader:
                idx_train = loader["idx_train"]
                idx_val = loader["idx_val"]
                idx_test = loader["idx_test"]
            return idx_train, idx_val, idx_test
        else:
            return super().get_train_val_test()

def deeprobust_to_pyG(
            dataset_name: str, 
            homoEdgeRatio: float, 
            dataset_index: int ) -> Data:
    """
    dataset_name: syn-cora or syn-products
    homoEdgeRatio: [0.00, 0.10, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 1.00]
    dataset_index: [1, 2, 3]
    """
    assert dataset_name in ['syn-cora', 'syn-products'], f'Unexpected dataset name {dataset_name}.'
    assert homoEdgeRatio in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 1.0], \
                    f'Please choos a homophily edge ratio in \[0.00, 0.10, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 1.00\]'
    name = 'h' + "{:.2f}".format(homoEdgeRatio) + '-' + 'r' + str(dataset_index)
    root = './data/'+ str(dataset_name)
    data = CustomDataset(root= str(root), name=str(name))
    adj, features, labels = data.adj, data.features, data.labels
    edge_index = torch.tensor(np.array(adj.nonzero()), dtype=torch.int64)
    x = torch.tensor(features.todense()).float()
    y = torch.tensor(labels).to(torch.int64)
    num_nodes = x.shape[0]
    num_features = x.shape[1]
    num_classes = int(max(y)) + 1
    return Data(x=x, edge_index=edge_index, y=y, num_nodes=num_nodes, num_features=num_features, num_classes=num_classes)

def split_syn_data(
        dataset_name: str, 
        homoEdgeRatio: float, 
        dataset_index: int,  
        split_type: str):
    """
    dataset_name: str, name of the synthetic dataset, either 'syn-cora' or 'syn-products'
                  c.f.: https://github.com/GemsLab/H2GCN/tree/master/npz-datasets
    homoEdgeRatio: float, homophily edge ratio
    dataset_index: int, in [1, 2, 3], 3 synthetic graphs are generated for each homophily edge ratio
    split_type: str, format {samples_in_one_fold}_{k fold}_{test_samples_per_class}. For example, 5_3f_85
                samples_in_one_fold: int, sample per fold ratio, sample x% of each class to one fold   
                k_fold: int, k-fold cross validation. One fold is used as validation the rest portions are used as training
                test_samples_per_class: int, test ratio, sample x% of each class for test set
    """
    assert dataset_name in ['syn-cora', 'syn-products'], f'Unexpected dataset name {dataset_name}.'
    assert homoEdgeRatio in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 1.0], \
                    f'Please choos a homophily edge ratio in \[0.00, 0.10, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 1.00\]'
    dataset = deeprobust_to_pyG(dataset_name, homoEdgeRatio, dataset_index)

    file_name = 'h' + "{:.2f}".format(homoEdgeRatio) + '-' + 'r' + str(dataset_index)
    raw_dir = Path(os.path.join('data','split', str(dataset_name), str(file_name), split_type))
    raw_dir.mkdir(parents=True, exist_ok=True)

    samples_in_one_fold = int(split_type.split("_")[0])
    k_fold = int(split_type.split("_")[1].replace("f",""))
    test_samples_per_class  = int(split_type.split("_")[2])

    # For each configuration we split the data five times
    for i in range(5):
        assert int(samples_in_one_fold)*int(k_fold)+int(test_samples_per_class) <= 100, "Invalid fraction" 
        k_fold_indices, test_indices = get_idx_split(dataset,
                    samples_per_class_in_one_fold=samples_in_one_fold/100.,
                    k_fold=k_fold,
                    test_samples_per_class=test_samples_per_class/100.)
        # print(k_fold_indices[0].shape)
        # print(k_fold_indices[1].shape)
        # print(k_fold_indices[2].shape)
        # print(k_fold_indices[3].shape)
        # print(k_fold_indices)
        # print(type(k_fold_indices))
        # print(test_indices.shape)
        split_file = f'{dataset_name.lower()}_{file_name}_split_{i}.npz'
        # print(f"sample/fold/test: {len(k_fold_indices[0])}/{len(k_fold_indices)}/{len(test_indices)}")
        # np.savez(raw_dir/split_file, k_fold_indices=k_fold_indices, test_indices=test_indices)
        np.savez(raw_dir/split_file, **{f'fold_{i}': arr for i, arr in enumerate(k_fold_indices)}, test_indices=test_indices)

def load_syn_data(
        dataset_name: str, 
        homoEdgeRatio: float, 
        dataset_index: int, 
        split_type: str, 
        split_ind: int,
        fold: int ) -> Data:
    """
    dataset_name: str, name of the synthetic dataset, either 'syn-cora' or 'syn-products'
                  c.f.: https://github.com/GemsLab/H2GCN/tree/master/npz-datasets
    homoEdgeRatio: float, homophily edge ratio
    dataset_index: int, in [1, 2, 3], 3 synthetic graphs are generated for each homophily edge ratio
    split_type: str, format {sample per fold ratio}_{k fold}_{test ratio}. For example, 5_3f_85
    split_ind: int, index of the split. In total five splits were generated for each dataset. 
    fold: int, index of the fold to be used as validation set. The rest k-1 folds will be used as training set.
    """
   
    assert dataset_name in ['syn-cora', 'syn-products'], f'Unexpected dataset name {dataset_name}.'
    assert homoEdgeRatio in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 1.0], \
                    f'Please choos a homophily edge ratio in \[0.00, 0.10, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 1.00\]'
    dataset = deeprobust_to_pyG(dataset_name, homoEdgeRatio, dataset_index)
    transform = NormalizeFeatures()
    dataset = transform(dataset)

    # load split files of masks of k-folds and test_indices 
    raw_file_name = 'h' + "{:.2f}".format(homoEdgeRatio) + '-' + 'r' + str(dataset_index)
    split_file_name = f'{dataset_name.lower()}_{raw_file_name}_split_{split_ind}.npz'
    full_path = os.path.join('data','split', str(dataset_name), str(raw_file_name), str(split_type), split_file_name)
    # print(full_path)
    # samples_in_one_fold = int(split_type.split("_")[0])
    # k_fold = int(split_type.split("_")[1].replace("f",""))
    # test_samples_per_class  = int(split_type.split("_")[2])
    
    if not os.path.exists(full_path):
        # print("Split data")
        split_syn_data(dataset_name, homoEdgeRatio, dataset_index, split_type)
        
    # print(f'load {full_path}')
    masks = np.load(full_path)
    
    val_fold_name = f'fold_{fold}'
    val_indices = masks[val_fold_name]
    arrays_to_concat = [arr for key, arr in masks.items() if key not in [val_fold_name, 'test_indices']]
    train_indices = np.concatenate(arrays_to_concat)
    # print(np.sort(train_indices))
    test_indices = masks['test_indices']

    dataset.train_mask = index_to_mask(torch.tensor(train_indices).flatten(), dataset.num_nodes)
    dataset.val_mask = index_to_mask(torch.tensor(val_indices).flatten(), dataset.num_nodes)
    dataset.test_mask = index_to_mask(torch.tensor(test_indices).flatten(), dataset.num_nodes)
    
    return dataset


def generate_node_to_nearest_training_4syndata(
                            dataset_name: str, 
                            homoEdgeRatio: float, 
                            dataset_index: int,  
                            split_type: str,
                            split_ind: int, 
                            fold: int,
                            bfs_depth = 10):
    max_split = 5
    max_fold = int(split_type.split("_")[1].replace("f",""))
    for split in tqdm(range(max_split)):
        raw_file_name = 'h' + "{:.2f}".format(homoEdgeRatio) + '-' + 'r' + str(dataset_index)
        split_file_name = f'{dataset_name.lower()}_{raw_file_name}_d2t_f{fold}.npz'
        raw_dir = Path(os.path.join('data','dist_to_train', str(dataset_name), raw_file_name, split_type))
        for fold in tqdm(range(max_fold)):
            data = load_syn_data(dataset_name, homoEdgeRatio, dataset_index, split_type, split_ind, fold)
            dist_to_train = torch.ones(data.num_nodes) * bfs_depth
            dist_to_train = shortest_path_length(data.edge_index, data.train_mask, bfs_depth)
            raw_split_dir = raw_dir / f'split_{split}'
            raw_split_dir.mkdir(parents=True, exist_ok=True)
            split_file = f'{dataset_name.lower()}_{raw_file_name}_d2t_f{fold}.npy'
            np.save(raw_split_dir/split_file, dist_to_train)
            
# Run at console -> python -c 'from data_utils import *; load_node_to_nearest_training('syn-cora',0, 1,'5_3f_85',0,0)'
def load_node_to_nearest_training_4syndata(
                            dataset_name: str, 
                            homoEdgeRatio: float, 
                            dataset_index: int,  
                            split_type: str, 
                            split_ind: int, 
                            fold: int):
    raw_file_name = 'h' + "{:.2f}".format(homoEdgeRatio) + '-' + 'r' + str(dataset_index)
    split_file_name = f'{dataset_name.lower()}_{raw_file_name}_d2t_f{fold}.npy'
    d2t_file = os.path.join('data', 'dist_to_train', str(dataset_name), raw_file_name, 
                               split_type, f'split_{split_ind}', split_file_name)
    if not os.path.isfile(d2t_file):
        generate_node_to_nearest_training_4syndata(dataset_name, homoEdgeRatio, dataset_index, split_type, split_ind, fold)
    return torch.from_numpy(np.load(d2t_file))

def compute_node_level_homophily(edge_index, y):
    edge_index = to_undirected(edge_index)
    
    homophily_scores = torch.zeros(y.size(0))
    bins = np.arange(0, 1.1, 0.1)
    groups = {i: [] for i in range(len(bins) - 1)}
    
    for node in range(y.size(0)):
        neighbors = edge_index[1][edge_index[0] == node]
        if len(neighbors) == 0:
            continue 
        
        node_label = y[node]
        neighbor_labels = y[neighbors]
        
        same_label_count = (neighbor_labels == node_label).sum().float()
        homophily_score = same_label_count / len(neighbors)
        
        homophily_scores[node] = homophily_score
        homophily_score_cpu = homophily_score.cpu().numpy()
        bin_index = np.digitize(homophily_score_cpu, bins) - 1
        if bin_index==10:
            bin_index = 9
        groups[bin_index].append(node)
        
    return homophily_scores, groups

def compute_degree_bins(edge_index):
    # 计算每个节点的度
    deg = edge_index[0].bincount(minlength=edge_index.max().item() + 1)
    
    # 根据度数排序节点
    sorted_nodes = torch.argsort(deg)
    
    # 初始化bins
    bins = {i: [] for i in range(20)}
    
    # 分配节点到相应的百分比bin中
    for idx, node_id in enumerate(sorted_nodes):
        # 计算节点的百分比位置
        percentile = (idx / len(sorted_nodes)) * 100
        # 确定百分比所在的bin
        bin_index = int(percentile // 5)
        bin_index = 19 if bin_index > 19 else bin_index  # 确保最大值落在最后一个bin中
        # 将节点添加到对应的bin中
        bins[bin_index].append(node_id.item())
        
    return bins