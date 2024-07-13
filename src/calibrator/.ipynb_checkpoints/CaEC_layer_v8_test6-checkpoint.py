from typing import Union, Optional
from torch_geometric.typing import OptPairTensor, Adj, OptTensor
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Parameter

from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax, degree, is_undirected
from src.data.data_utils import shortest_path_length
from src.model.model import MLP
from torch_sparse import SparseTensor, matmul
import scipy.sparse


class CaEC_Layer(MessagePassing):
    _alpha: OptTensor

    def __init__(
            self,
            in_channels: int,
            hidden_channels: int,
            out_channels: int,
            edge_index: Adj,
            num_nodes: int,
            num_class: int,
            num_layers: int = 2,
            **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.edge_index = edge_index
        self.num_nodes = num_nodes
        self.num_class = num_class
        self.undirected = is_undirected(edge_index)

        self.mlp = MLP(2*in_channels, hidden_channels, 1, num_layers=2, dropout=0)
        # self.lin = torch.nn.Linear(num_class, 1)
        # self.W = Parameter(torch.ones(num_nodes,1))
        self.bias = Parameter(torch.ones(1))
        self.epsilon = Parameter(torch.zeros(2))

        
        if self.undirected:
            self.degrees = degree(edge_index[1, :], num_nodes)
            self.max_degree = self.degrees.max()
        else:
            self.in_degrees = degree(edge_index[1, :], num_nodes)
            self.out_degrees = degree(edge_index[0, :], num_nodes)
            self.max_in_degrees = self.in_degrees.max()
            self.max_out_degrees = self.out_degrees.max()

        self.reset_parameters()
           
            
    def reset_parameters(self):
        self.mlp.reset_parameters()
        # self.W.reset_parameters()

    def forward(self, x: Union[Tensor, OptPairTensor], flag=False):
        # if flag:
        #     dropout_rates = self.dynamic_EdgeDropout()
        #     random_probs = torch.rand(self.edge_index.size(1)).to(x[0].device)
        #     retained_edges = random_probs > dropout_rates
        #     edge_index_dropout = self.edge_index[:, retained_edges]
            
        probs = F.softmax(x, dim=1)
        logprobs = torch.log(probs)
        eps = F.softmax(self.epsilon, dim=-1)
        
       
        if self.undirected:
            deg = self.degrees
        else:
            deg = self.out_degrees
        deg_inverse = 1 / torch.sqrt(deg)
        deg_inverse[deg_inverse == float('inf')] = 0
        
        out = self.propagate(edge_index=self.edge_index, h=x*deg_inverse.unsqueeze(-1))
        out_L = eps[0]*x + out
        out_H = eps[1]*x - out
        
        # Concatenate
        out = self.mlp(torch.cat([out_L, out_H], dim=1), input_tensor=True)
        # Weighted sum
        # out = out_L + out_H
        # out = self.lin(out)
        
        # out = w[0]*out + w[1]*out2
        # print('out.shape:',out.shape)
        out = F.softplus(out) + F.softplus(self.bias)
        # out = F.softplus(out)+ w *out + (1-w)*out2
        # 
        # print(out.shape)
        return out

        
    # def message(
    #         self,
    #         h_j: Tensor,
    #         probs_i: Tensor,
    #         logprobs_j: Tensor,
    #         epsilon: Tensor,
    #         index: Tensor,
    #         ptr: OptTensor,
    #         size_i: Optional[int]) -> Tensor:
    #     """
    #     h: [N, hidden_channels]
    #     """
    #     # print('logprobs_i.shape:', logprobs_i.shape)
    #     # print('probs_i.shape:', probs_j.shape)
    #     kl_div = F.kl_div(logprobs_j, probs_i, reduction='none')
    #     kl_div_per_node = torch.sum(kl_div, dim=1)
    #     weights = torch.exp(-kl_div_per_node)
    #     # print(kl_div_per_node.shape)
    #     weights = softmax(weights, index, ptr, size_i)
    #     return h_j * weights.unsqueeze(-1).expand_as(h_j)

    def message(
            self,
            h_j: Tensor,
            index: Tensor,
            ptr: OptTensor,
            size_i: Optional[int]) -> Tensor:
        """
        h: [N, hidden_channels]
        """
        # print('logprobs_i.shape:', logprobs_i.shape)
        # print('probs_i.shape:', probs_j.shape)
       
        return h_j
   