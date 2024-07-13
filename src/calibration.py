import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import abc
import torch
from torch import Tensor, LongTensor
import torch.nn.functional as F
import gc
import math
import time
from pathlib import Path
from torch_geometric.utils import remove_self_loops
from src.data.data_utils import load_data, load_node_to_nearest_training
from src.model.model import create_model
from src.calibrator.calibrator import \
    TS, VS, ETS, CaGCN, GATS, IRM, SplineCalib, Dirichlet, OrderInvariantCalib, ASTS
from src.calibloss import \
    NodewiseECE, NodewiseBrier, NodewiseNLL, Reliability, NodewiseKDE, \
    NodewiswClassECE
from src.utils import \
    set_global_seeds, arg_parse, name_model, create_nested_defaultdict, \
    metric_mean, metric_std, default_cal_wdecay, save_prediction


# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"



# collects metrics for evaluation
class Metrics(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def acc(self) -> float:
        raise NotImplementedError

    @abc.abstractmethod
    def nll(self) -> float:
        raise NotImplementedError

    @abc.abstractmethod
    def brier(self) -> float:
        raise NotImplementedError

    @abc.abstractmethod
    def ece(self) -> float:
        raise NotImplementedError

    @abc.abstractmethod
    def reliability(self) -> Reliability:
        raise NotImplementedError

    @abc.abstractmethod
    def kde(self) -> float:
        raise NotImplementedError

    @abc.abstractmethod
    def cls_ece(self) -> float:
        raise NotImplementedError

class NodewiseMetrics(Metrics):
    def __init__(
            self, logits: Tensor, gts: LongTensor, index: LongTensor,
            bins: int = 15, scheme: str = 'equal_width', norm=1):
        self.node_index = index
        self.logits = logits
        self.gts = gts
        self.nll_fn = NodewiseNLL(index)
        self.brier_fn = NodewiseBrier(index)
        self.ece_fn = NodewiseECE(index, bins, scheme, norm)
        self.kde_fn = NodewiseKDE(index, norm)
        self.cls_ece_fn = NodewiswClassECE(index, bins, scheme, norm)

    def acc(self) -> float:
        preds = torch.argmax(self.logits, dim=1)[self.node_index]
        return torch.mean(
            (preds == self.gts[self.node_index]).to(torch.get_default_dtype())
        ).item()

    def nll(self) -> float:
        return self.nll_fn(self.logits, self.gts).item()

    def brier(self) -> float:
        return self.brier_fn(self.logits, self.gts).item()

    def ece(self) -> float:
        return self.ece_fn(self.logits, self.gts).item()

    def reliability(self) -> Reliability:
        return self.ece_fn.get_reliability(self.logits, self.gts)
    
    def kde(self) -> float:
        return self.kde_fn(self.logits, self.gts).item()

    def cls_ece(self) -> float:
        return self.cls_ece_fn(self.logits, self.gts).item()


def eval(data, log_prob, mask_name):
    if mask_name == 'Train':
        mask = data.train_mask
    elif mask_name == 'Val':
        mask = data.val_mask
    elif mask_name == 'Test':
        mask = data.test_mask
    else:
        raise ValueError("Invalid mask_name")
    eval_result = {}
    eval = NodewiseMetrics(log_prob, data.y, mask)
    acc, nll, brier, ece, kde, cls_ece = eval.acc(), eval.nll(), \
                                eval.brier(), eval.ece(), eval.kde(), eval.cls_ece()
    eval_result.update({'acc':acc,
                        'nll':nll,
                        'bs':brier,
                        'ece':ece,
                        'kde':kde,
                        'cls_ece': cls_ece})
    reliability = eval.reliability()
    del eval
    gc.collect()
    return eval_result, reliability


def main(dataset_index, split, init, eval_type_list, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    uncal_test_result = create_nested_defaultdict(eval_type_list)
    cal_val_result = create_nested_defaultdict(eval_type_list)
    cal_test_result = create_nested_defaultdict(eval_type_list)
    max_fold = int(args.split_type.split("_")[1].replace("f",""))

    for fold in range(max_fold):
        # Load data
        
        dataset = load_data(args.dataset, args.split_type, split, fold)
        data = dataset._data.to(device)
        num_classes = dataset.num_classes
       
        if args.remove_selfloop == True:
            data.edge_index, _ = remove_self_loops(data.edge_index)

        # Load model
        model = create_model(data, args).to(device)
        model_name = name_model(fold, args)
        if args.dataset in ['syn-cora','syn-products']:
            raw_file_name = 'h' + "{:.2f}".format(args.homoEdgeRatio) + '-' + 'r' + str(dataset_index)
            dir = Path(os.path.join('model', args.dataset, str(raw_file_name), args.split_type, 'split'+str(split), 'init'+ str(init)))
        else:
            dir = Path(os.path.join('model', args.dataset, args.split_type, 'split'+str(split), 'init'+ str(init)))
        file_name = dir / (model_name + '.pt')
        model.load_state_dict(torch.load(file_name))
        torch.cuda.empty_cache()

        with torch.no_grad():
            model.eval()
            logits = model(data)
            log_prob = logits
        # print("Uncal logits:", logits)
        # print("Uncal log_prob:", log_prob)
        ### Store uncalibrated result
        if args.save_prediction:
            save_prediction(log_prob.cpu().numpy(), args.dataset, args.split_type, split, init, fold, args.model, "uncal")

        for eval_type in eval_type_list:
            eval_result, reliability = eval(data, log_prob, 'Test')
            # print("Uncal reliability:", reliability)
            for metric in eval_result:
                uncal_test_result[eval_type][metric].append(eval_result[metric])
        torch.cuda.empty_cache()

        ### Calibration
        if args.calibration == 'TS':
            temp_model = TS(model)
        elif args.calibration == 'IRM':
            temp_model = IRM(model)
        elif args.calibration == 'Spline':
            temp_model = SplineCalib(model, 7)
        elif args.calibration == 'Dirichlet':
            temp_model = Dirichlet(model, num_classes)
        elif args.calibration == 'OrderInvariant':
            temp_model = OrderInvariantCalib(model, num_classes)
        elif args.calibration == 'VS':
            temp_model = VS(model, num_classes)
        elif args.calibration == 'ETS':
            temp_model = ETS(model, num_classes)
        elif args.calibration == 'CaGCN':
            temp_model = CaGCN(model, data.num_nodes, num_classes, args.cal_dropout_rate)
        elif args.calibration == 'GATS':
            dist_to_train = load_node_to_nearest_training(args.dataset,  args.split_type, split, fold)
            temp_model = GATS(model, data.edge_index, data.num_nodes, data.train_mask,
                            num_classes, dist_to_train, args.gats_args)
        elif args.calibration == 'ASTS':
            temp_model = ASTS(model, data.edge_index, data.num_nodes, num_classes, args.hidden_channels_ASTS)

        ### Train the calibrator on validation set and validate it on the training set
        cal_wdecay = args.cal_wdecay if args.cal_wdecay is not None else default_cal_wdecay(args)
        lr = args.cal_lr if args.cal_lr is not None else 0.01
        temp_model.fit(data, data.val_mask, data.train_mask, cal_wdecay, lr)
        with torch.no_grad():
            temp_model.eval()
            logits = temp_model(data)
            log_prob = logits

        # Store calibrated result
        if args.save_prediction:
            save_prediction(log_prob.cpu().numpy(), args.dataset, args.split_type, split, init, fold, args.model, args.calibration)
        # print("logits:", logits)
        # print("Calibrated log_prob:", log_prob)
        ### The training set is the validation set for the calibrator
        for eval_type in eval_type_list:
            eval_result, _ = eval(data, log_prob, 'Train')
            for metric in eval_result:
                cal_val_result[eval_type][metric].append(eval_result[metric])

        for eval_type in eval_type_list:
            eval_result, reliability = eval(data, log_prob, 'Test')
            # print('ece: ', eval_result['ece'])
            # print("Cal reliability:", reliability)
            for metric in eval_result:
                cal_test_result[eval_type][metric].append(eval_result[metric])
                # if math.isinf(eval_result['nll']):
                #     print(logits)
                #     print(data.y[data.test_mask])
                #     nll_temp = F.cross_entropy(log_prob[data.test_mask], data.y[data.test_mask])
                #     print('nll_temp:', nll_temp)
        
                # if eval_result['nll']>100:
                #     print(logits)
                #     # print(data.y[data.test_mask])
                #     nll_temp = F.cross_entropy(log_prob[data.test_mask], data.y[data.test_mask])
                #     print('nll_temp:', nll_temp)
        # print('Reliability: ', reliability)
        
        torch.cuda.empty_cache()
    return uncal_test_result, cal_val_result, cal_test_result


if __name__ == '__main__':
    start_time = time.time()
    args = arg_parse()
    print(args)
    set_global_seeds(args.seed)
    eval_type_list = ['Nodewise']
    max_splits,  max_init = 5, 5
    num_datasets = 1
    if args.dataset in ['syn-cora','syn-products']:
        num_datasets = 3

    uncal_test_total = create_nested_defaultdict(eval_type_list)
    cal_val_total = create_nested_defaultdict(eval_type_list)
    cal_test_total = create_nested_defaultdict(eval_type_list)
    for dataset_index in range(num_datasets):
        for split in range(max_splits):
            for init in range(max_init):
                print(split, init)
                (uncal_test_result,
                 cal_val_result,
                 cal_test_result) = main(int(dataset_index+1), split, init, eval_type_list, args)
                # print(f'uncal_test_nll: {uncal_test_result[eval_type]['nll']}')
                # print(f'cal_test_kde: {cal_test_result[eval_type]['nll']}')
                for eval_type, eval_metric in uncal_test_result.items():
                    for metric in eval_metric:
                        uncal_test_total[eval_type][metric].extend(uncal_test_result[eval_type][metric])
                        cal_val_total[eval_type][metric].extend(cal_val_result[eval_type][metric])
                        cal_test_total[eval_type][metric].extend(cal_test_result[eval_type][metric])

    val_mean = metric_mean(cal_val_total['Nodewise'])
    # validate calibrator
    print(f"Val NNL: &{val_mean['nll']:.4f}")
    # print results

    for name, result in zip([args.calibration], [cal_val_total]):
        print('Validation results:', name)
        for eval_type in eval_type_list:
            # print(type(result))
            val_mean = metric_mean(result[eval_type])
            val_std = metric_std(result[eval_type])
            # print(f"{eval_type:>8} Accuracy: &{test_mean['acc']:.2f}$\pm${test_std['acc']:.2f} \t" + \
            #                     f"NLL: &{test_mean['nll']:.4f}$\pm${test_std['nll']:.4f} \t" + \
            #                     f"Brier: &{test_mean['bs']:.4f}$\pm${test_std['bs']:.4f} \t" + \
            #                     f"ECE: &{test_mean['ece']:.2f}$\pm${test_std['ece']:.2f} \t" + \
            #                     f"Classwise-ECE: &{test_mean['cls_ece']:.2f}$\pm${test_std['cls_ece']:.2f} \t" + \
            #                     f"KDE: &{test_mean['kde']:.2f}$\pm${test_std['kde']:.2f}")
            print("----------" +  "\t" +  f"{args.model}\t" + f"{args.dataset}\t" + f"{args.calibration}\t" + "----------")
            print(f"{name}" + " " * 6 + f"{eval_type:>8} Accuracy" + " " * 5 + "NLL" + " " * 8 + "Brier" + \
                  " " * 7 + "ECE" + " " * 5 + "Classwise-ECE" + " " * 5 + "KDE")
            # print("mean" + " " * 7 + f"{test_mean['acc']:.2f}\t" + f"{test_mean['nll']:.4f}\t" + f"{test_mean['bs']:.4f}\t" + \
            #       f"{test_mean['ece']:.2f}\t" + f"{test_mean['cls_ece']:.2f}\t" + f"{test_mean['kde']:.2f}")
            # print("std" + " " * 8 + f"{test_std['acc']:.2f}\t" + f"{test_std['nll']:.4f}\t " + f"{test_std['bs']:.4f}\t " + \
            #       f"{test_std['ece']:.2f}\t" + f"{test_std['cls_ece']:.2f}\t" + f"{test_std['kde']:.2f}")
            print("mean " + " \t" + f"{val_mean['acc']:.2f}\t" + f"{val_mean['nll']:.4f}\t" + f"{val_mean['bs']:.4f}\t" + \
                  f"{val_mean['ece']:.2f}\t" + f"{val_mean['cls_ece']:.2f}\t" + f"{val_mean['kde']:.2f}")
            print("std" + " \t" + f"{val_std['acc']:.2f}\t" + f"{val_std['nll']:.4f}\t " + f"{val_std['bs']:.4f}\t " + \
                  f"{val_std['ece']:.2f}\t" + f"{val_std['cls_ece']:.2f}\t" + f"{val_std['kde']:.2f}")
            
    for name, result in zip(['Uncal', args.calibration], [uncal_test_total, cal_test_total]):
        print('Test results:', name)
        for eval_type in eval_type_list:
            # print(cal_test_total[eval_type]['nll'])
            test_mean = metric_mean(result[eval_type])
            test_std = metric_std(result[eval_type])
            # print(f"{eval_type:>8} Accuracy: &{test_mean['acc']:.2f}$\pm${test_std['acc']:.2f} \t" + \
            #                     f"NLL: &{test_mean['nll']:.4f}$\pm${test_std['nll']:.4f} \t" + \
            #                     f"Brier: &{test_mean['bs']:.4f}$\pm${test_std['bs']:.4f} \t" + \
            #                     f"ECE: &{test_mean['ece']:.2f}$\pm${test_std['ece']:.2f} \t" + \
            #                     f"Classwise-ECE: &{test_mean['cls_ece']:.2f}$\pm${test_std['cls_ece']:.2f} \t" + \
            #                     f"KDE: &{test_mean['kde']:.2f}$\pm${test_std['kde']:.2f}")
            print("----------" +  "\t" +  f"{args.model}\t" + f"{args.dataset}\t" + f"{args.calibration}\t" + "----------")
            print(f"{name}" + " " * 6 + f"{eval_type:>8} Accuracy" + " " * 5 + "NLL" + " " * 8 + "Brier" + \
                  " " * 7 + "ECE" + " " * 5 + "Classwise-ECE" + " " * 5 + "KDE")
            # print("mean" + " " * 7 + f"{test_mean['acc']:.2f}\t" + f"{test_mean['nll']:.4f}\t" + f"{test_mean['bs']:.4f}\t" + \
            #       f"{test_mean['ece']:.2f}\t" + f"{test_mean['cls_ece']:.2f}\t" + f"{test_mean['kde']:.2f}")
            # print("std" + " " * 8 + f"{test_std['acc']:.2f}\t" + f"{test_std['nll']:.4f}\t " + f"{test_std['bs']:.4f}\t " + \
            #       f"{test_std['ece']:.2f}\t" + f"{test_std['cls_ece']:.2f}\t" + f"{test_std['kde']:.2f}")
            print("mean " + " \t" + f"{test_mean['acc']:.2f}\t" + f"{test_mean['nll']:.4f}\t" + f"{test_mean['bs']:.4f}\t" + \
                  f"{test_mean['ece']:.2f}\t" + f"{test_mean['cls_ece']:.2f}\t" + f"{test_mean['kde']:.2f}")
            print("std" + " \t" + f"{test_std['acc']:.2f}\t" + f"{test_std['nll']:.4f}\t " + f"{test_std['bs']:.4f}\t " + \
                  f"{test_std['ece']:.2f}\t" + f"{test_std['cls_ece']:.2f}\t" + f"{test_std['kde']:.2f}")
    end_time = time.time()
    running_time = end_time - start_time
    print(f"Running time: {running_time} seconds")