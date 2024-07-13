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


class ASTS_Layer(MessagePassing):
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
        self.alpha_dd = Parameter(torch.zeros(1))
        self.beta_dd = Parameter(torch.zeros(1)) 
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
        if flag:
            dropout_rates = self.dynamic_EdgeDropout()
            random_probs = torch.rand(self.edge_index.size(1)).to(x[0].device)
            retained_edges = random_probs > dropout_rates
            edge_index_dropout = self.edge_index[:, retained_edges]
            
        probs = F.softmax(x, dim=1)
        logprobs = torch.log(probs)
        eps = F.softmax(self.epsilon, dim=-1)
        
        
        if flag:
            row, col = edge_index_dropout
            deg = degree(edge_index_dropout[1, :], self.num_nodes)
            deg_inverse = 1 / torch.sqrt(deg)
            deg_inverse[deg_inverse == float('inf')] = 0
            norm = deg_inverse[row] * deg_inverse[col]
            
            out = self.propagate(edge_index=edge_index_dropout, h=x, norm=norm) 
            out_L = eps[0]*x + out
            out_H = eps[1]*x - out
        else:
            if self.undirected:
                deg = self.degrees
            else:
                deg = self.out_degrees

            row, col = self.edge_index
            deg_inverse = 1 / torch.sqrt(deg)
            deg_inverse[deg_inverse == float('inf')] = 0
            norm = deg_inverse[row] * deg_inverse[col]
            
            out = self.propagate(edge_index=self.edge_index, h=x, norm=norm)
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

    
    def dynamic_EdgeDropout(self):
        alpha_dd = F.softplus(self.alpha_dd)
        beta_dd = F.softplus(self.beta_dd - 1)
        if self.undirected:
            degrees_u = self.degrees[self.edge_index[0, :]]
            degrees_v = self.degrees[self.edge_index[1, :]]
            dropout_rates = alpha_dd * torch.pow((degrees_u * degrees_v) / (self.max_degree ** 2), beta_dd)
            # dropout_rates = torch.sigmoid(self.alpha_dd * torch.pow((degrees_u * degrees_v) / (self.max_degree ** 2 ), self.beta_dd)+self.eta_dd)
        else:
            out_degrees_u = self.out_degrees[self.edge_index[0, :]]
            in_degrees_v = self.in_degrees[self.edge_index[1, :]]
            dropout_rates = alpha_dd * torch.pow((in_degrees_v * out_degrees_u) / (self.max_in_degrees * self.max_out_degrees) , beta_dd)

        dropout_rates = torch.sigmoid(dropout_rates)
        return dropout_rates 

    # def init_adj(self, edge_index):
    #     """ cache normalized adjacency and normalized strict two-hop adjacency,
    #     neither has self loops
    #     """
        
    #     row, col = edge_index
    #     adj_t = SparseTensor(row=col, col=row, value=None, sparse_sizes=(self.num_nodes, self.num_nodes))

    #     adj_t.remove_diag(0)
    #     adj_t2 = matmul(adj_t, adj_t)
    #     adj_t2.remove_diag(0)
    #     adj_t = scipy.sparse.csr_matrix(adj_t.to_scipy())
    #     adj_t2 = scipy.sparse.csr_matrix(adj_t2.to_scipy())
    #     adj_t2 = adj_t2 - adj_t
    #     adj_t2[adj_t2 > 0] = 1
    #     adj_t2[adj_t2 < 0] = 0

    #     adj_t2 = SparseTensor.from_scipy(adj_t2)
    #     adj_t2 = adj_t2.to(edge_index.device)
    #     row, col, _ = adj_t2.coo()
    #     return torch.stack([row, col], dim=0) 
        
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
            norm: Tensor,
            index: Tensor,
            ptr: OptTensor,
            size_i: Optional[int]) -> Tensor:
        """
        h: [N, hidden_channels]
        """
        # print('logprobs_i.shape:', logprobs_i.shape)
        # print('probs_i.shape:', probs_j.shape)
       
        return norm.view(-1, 1) * h_j
   