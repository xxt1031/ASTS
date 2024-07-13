import os
import math
import random
import argparse
import torch
import yaml
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from typing import Sequence
from calibloss import ECE, Reliability

def set_global_seeds(seed):
    """
    Set global seed for reproducibility
    """  
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
    except ImportError:
        pass

    np.random.seed(seed)
    random.seed(seed)

def arg_parse():
    parser = argparse.ArgumentParser(description='train.py and calibration.py share the same arguments')
    parser.add_argument('--seed', type=int, default=10, help='Random Seed')
    parser.add_argument('--dataset', type=str, default='genius', choices=['syn-cora','syn-products', 'genius', 'twitch-gamer', 'Cora','Citeseer',
                    'Pubmed', 'Computers', 'Photo', 'CS', 'Physics', 'CoraFull','Texas', 'Wisconsin', 'Actor', 'squirrel', 'chameleon', 'Cornell'])
    parser.add_argument('--split_type', type=str, default='20_3f_40', help='k-fold and test split')
    parser.add_argument('--model', type=str, default='GCN', choices=['GCN', 'GAT', 'MLP', 'LINK', 'LINK_Concat', 'LINKX', 'H2GCN'])
    parser.add_argument('--verbose', action='store_true', default=False, help='Show training and validation loss')
    parser.add_argument('--wdecay', type=float, default=5e-4, help='Weight decay for training phase')
    parser.add_argument('--dropout_rate', type=float, default=0.5, help='Dropout rate. 1.0 denotes drop all the weights to zero')
    parser.add_argument('--calibration', type=str, default='ASTS',  help='Post-hoc calibrators')
    parser.add_argument('--cal_wdecay', type=float, default=None, help='Weight decay for calibration phase')
    parser.add_argument('--cal_lr', type=float, default=None, help='Learning rate for calibration phase')
    parser.add_argument('--cal_dropout_rate', type=float, default=0.5, help='Dropout rate for calibrators (CaGCN)')
    parser.add_argument('--folds', type=int, default=3, help='K folds cross-validation for calibration')
    parser.add_argument('--ece-bins', type=int, default=15, help='number of bins for ece')
    parser.add_argument('--ece-scheme', type=str, default='equal_width', choices=ECE.binning_schemes, help='binning scheme for ece')
    parser.add_argument('--ece-norm', type=float, default=1.0, help='norm for ece')
    parser.add_argument('--save_prediction', action='store_true', default=False)
    parser.add_argument('--remove_selfloop', action='store_true', default=False)
    parser.add_argument('--num_mlp_layers', type=int, default=1, help='number of mlp layers in h2gcn')
    parser.add_argument('--hidden_channels_ASTS', type=int, default=16, help='dimension of hidden layers in the MLP of ASTS')
    parser.add_argument('--config', action='store_true', default=False)

    gats_parser = parser.add_argument_group('optional GATS arguments')
    gats_parser.add_argument('--heads', type=int, default=8, help='Number of heads for GATS. Hyperparameter set: {1,2,4,8,16}')
    gats_parser.add_argument('--bias', type=float, default=1, help='Bias initialization for GATS')
    gats_parser.add_argument('--dropout_rate_gats', type=float, default=0, help='Dropout rate for GATS')

    linkx_parser = parser.add_argument_group('optional LINKX arguments')
    linkx_parser.add_argument('--inner_activation', action='store_true', default=False, help='Whether linkV3 uses inner activation')
    linkx_parser.add_argument('--inner_dropout', action='store_true', default=False, help='Whether linkV3 uses inner dropout')
    linkx_parser.add_argument('--link_init_layers_A', type=int, default=1)
    linkx_parser.add_argument('--link_init_layers_X', type=int, default=1)
    args = parser.parse_args()
    if args.config:
        config = read_config(args)
        for key, value in config.items():
            setattr(args, key, value)

    args_dict = {}
    for group in parser._action_groups:
        if group.title == 'optional GATS arguments':
            group_dict={a.dest:getattr(args,a.dest,None) for a in group._group_actions}
            args_dict['gats_args'] = argparse.Namespace(**group_dict)
        elif group.title == 'optional LINKX arguments':
            group_dict={a.dest:getattr(args,a.dest,None) for a in group._group_actions}
            args_dict['linkx_args'] = argparse.Namespace(**group_dict)
        else:
            group_dict={a.dest:getattr(args,a.dest,None) for a in group._group_actions}
            args_dict.update(group_dict)
    return argparse.Namespace(**args_dict)

def read_config(args):
    dir = Path(os.path.join('config', args.calibration))
    file_name = f'{args.dataset}_{args.model}.yaml'
    try:
        with open(dir/file_name) as file:
            yaml_file = yaml.safe_load(file)
    except IOError:
        yaml_file = {}
    if yaml_file is None:
        yaml_file = {}
    return yaml_file

def default_cal_wdecay(args):
    if args.calibration in ['TS', 'VS', 'ETS']:
        return 0
    elif args.calibration == 'CaGCN':
        if args.dataset == "CoraFull":
            return 0.03
        else:
            return 5e-3
    else:
        return 5e-4

def name_model(fold, args):
    assert args.model in ['GCN', 'GAT', 'MLP', 'LINK', 'LINK_Concat', 'LINKX', 'H2GCN'], f'Unexpected model name {args.model}.'
    name = args.model
    name += "_dp" + str(args.dropout_rate).replace(".","_") + "_"
    try:
        power =-math.floor(math.log10(args.wdecay))
        frac = str(args.wdecay)[-1] if power <= 4 else str(args.wdecay)[0]
        name += frac + "e_" + str(power)
    except:
        name += "0"
    name += "_f" + str(fold)
    return name

def metric_mean(result_dict):
    out = {}
    for key, val in result_dict.items():
        if key in ['acc', 'ece', 'cls_ece', 'kde']:
            weight = 100
        else:
            weight = 1
        out[key] = np.mean(val) * weight
    return out

def metric_std(result_dict):
    out = {}
    for key, val in result_dict.items():
        if key in ['acc', 'ece', 'cls_ece', 'kde']:
            weight = 100
        else:
            weight = 1
        out[key] = np.sqrt(np.var(val)) * weight
    return out

def create_nested_defaultdict(key_list):
    # To do: extend to *args
    out = {}
    for key in key_list:
        out[key] = defaultdict(list)
    return out

def save_prediction(predictions, name, split_type, split, init, fold, model, calibration):
    raw_dir = Path(os.path.join('predictions', model, str(name), calibration.lower(), split_type))
    raw_dir.mkdir(parents=True, exist_ok=True)
    file_name = f'split{split}' + f'init{init}' + f'fold{fold}' + '.npy'
    np.save(raw_dir/file_name, predictions)

def load_prediction(name, split_type, split, init, fold, model, calibration):
    raw_dir = Path(os.path.join('predictions', model, str(name), calibration.lower(), split_type))
    file_name = f'split{split}' + f'init{init}' + f'fold{fold}' + '.npy'
    return np.load(raw_dir / file_name)

def plot_reliabilities(
        reliabilities: Sequence[Reliability], args, uncal=True):
    linewidth = 1.0

    confs = [(r[0] / (r[2] + torch.finfo().tiny)).cpu().numpy()
             for r in reliabilities]
    accs = [(r[1] / (r[2] + torch.finfo().tiny)).cpu().numpy()
            for r in reliabilities]
    masks = [r[2].cpu().numpy() > 0 for r in reliabilities]

    nonzero_counts = np.sum(np.asarray(masks, dtype=np.int64), axis=0)
    conf_mean = np.sum(
        np.asarray(confs), axis=0) / (nonzero_counts + np.finfo(float).tiny)
    acc_mean = np.sum(
        np.asarray(accs), axis=0) / (nonzero_counts + np.finfo(float).tiny)
    acc_std = np.sqrt(
        np.sum(np.asarray(accs) ** 2, axis=0)
        / (nonzero_counts + np.finfo(float).tiny)
        - acc_mean ** 2)
    conf_mean = conf_mean[nonzero_counts > 0]
    acc_mean = acc_mean[nonzero_counts > 0]
    acc_std = acc_std[nonzero_counts > 0]

    fig, ax1 = plt.subplots(figsize=(2, 2), facecolor='w')
    for conf, acc, mask in zip(confs, accs, masks):
        ax1.plot(
            conf[mask], acc[mask], color='lightgray',
            linewidth=linewidth / 2.0, zorder=0.0)
    ax1.plot(
        [0, 1], [0, 1], color='black', linestyle=':', linewidth=linewidth,
        zorder=0.8)
    ax1.plot(
        conf_mean, acc_mean, color='blue', linewidth=linewidth, zorder=1.0)
    ax1.fill_between(
        conf_mean, acc_mean - acc_std, acc_mean + acc_std, facecolor='b',
        alpha=0.3, zorder=0.9)

    ax1.set_xlabel("Confidence")
    ax1.set_ylabel("Accuracy")
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    # ax1.legend(loc="lower right")
    
    plt.tight_layout()
    ax1.set_aspect(1)
    if uncal == True:
        figure_path = Path(os.path.join('figure', 'confidence_diagram', 'uncal'))
        ax1.set_title(f'Uncal-{args.model}-{args.dataset}')
        figure_filename = f'Uncal-{args.model}-{args.dataset}'
    else:
        figure_path = Path(os.path.join('figure', 'confidence_diagram', args.calibration))
        ax1.set_title(f'{args.calibration}-{args.model}-{args.dataset}')
        figure_filename = f'{args.calibration}-{args.model}-{args.dataset}'
    plt.savefig(figure_path/figure_filename, bbox_inches='tight', pad_inches=0)

    # Displaying the chart
    plt.show()
    plt.close() 