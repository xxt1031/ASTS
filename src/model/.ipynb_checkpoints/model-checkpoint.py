import torch
import torch.nn.functional as F
from torch.nn import Linear, Dropout
from torch_geometric.nn import GCNConv, GATConv, JumpingKnowledge
from torch_geometric.nn.conv.gcn_conv import gcn_norm
# from torch_geometric.nn import GCNConv, SGConv, GATConv, JumpingKnowledge, APPNP, GCN2Conv, MessagePassing
from torch_sparse import SparseTensor, matmul
import scipy.sparse

# from dgl import function as fn
# import numpy as np

def create_model(dataset, args):
    """
    Create model with hyperparameters
    """  

    num_layers = 2
    if args.model == 'GAT':
        num_hidden = 8
        attention_head = [8, 1]
    else:
        num_hidden = 64

    num_nodes = dataset.x.shape[0]
    num_classes = int(max(dataset.y))+1
    num_features = dataset.x.shape[1]
    if args.model == 'GCN':
        return GCN(in_channels=num_features, num_classes=num_classes, num_hidden=num_hidden,
                    drop_rate=args.dropout_rate, num_layers=num_layers)
    elif args.model == 'GAT':
        return GAT(in_channels=num_features, num_classes = num_classes, num_hidden=num_hidden,
                    attention_head=attention_head, drop_rate=args.dropout_rate, num_layers=num_layers)
    elif args.model == 'MLP':
        return MLP(in_channels=num_features, hidden_channels=num_hidden, out_channels = num_classes, num_layers=num_layers,
                    dropout=args.dropout_rate)
    elif args.model == 'H2GCN':
        return H2GCN(in_channels=num_features, hidden_channels=num_hidden, out_channels=num_classes, edge_index=dataset.edge_index,
                        num_nodes=num_nodes, num_layers=num_layers, dropout=args.dropout_rate,
                        num_mlp_layers=args.num_mlp_layers)
    elif args.model == 'LINK':
        return LINK(num_nodes=num_nodes, out_channels=num_classes)
    elif args.model == 'LINK_Concat':
        return LINK_Concat(in_channels=num_features, hidden_channels=num_hidden, out_channels=num_classes, num_layers=num_layers, num_nodes=num_nodes, dropout=args.dropout_rate)
    elif args.model == 'LINKX':
        return LINKX(in_channels=num_features, hidden_channels=num_hidden, out_channels=num_classes, num_layers=num_layers, num_nodes=num_nodes, dropout=args.dropout_rate, linkx_args=args.linkx_args)



class GCN(torch.nn.Module):
    def __init__(self, in_channels, num_classes, num_hidden, drop_rate, num_layers):
        super().__init__()
        self.drop_rate = drop_rate
        self.feature_list = [in_channels, num_hidden, num_classes]
        for _ in range(num_layers-2):
            self.feature_list.insert(-1, num_hidden)
        layer_list = []

        for i in range(len(self.feature_list)-1):
            layer_list.append(["conv"+str(i+1), GCNConv(self.feature_list[i], self.feature_list[i+1])])
        
        self.layer_list = torch.nn.ModuleDict(layer_list)

    def forward(self,data):
        x = data.x
        edge_index = data.edge_index
        for i in range(len(self.feature_list)-1):
            x = self.layer_list["conv"+str(i+1)](x, edge_index)
            if i < len(self.feature_list)-2:
                x = F.relu(x)
                x = F.dropout(x, self.drop_rate, self.training)
        return x

class GAT(torch.nn.Module):
    def __init__(self, in_channels, num_classes, num_hidden, attention_head, drop_rate, num_layers):
        super().__init__()
        self.drop_rate = drop_rate
        self.feature_list = [in_channels, num_hidden, num_classes]
        for _ in range(num_layers-2):
            self.feature_list.insert(-1, num_hidden)
        attention_head = [1] + attention_head
        layer_list = []
        for i in range(len(self.feature_list)-1):
            concat = False if i == num_layers-1 else True 
            layer_list.append(["conv"+str(i+1), GATConv(self.feature_list[i]* attention_head[i], self.feature_list[i+1], 
                                                        heads=attention_head[i+1], dropout=drop_rate, concat=concat)])
        self.layer_list = torch.nn.ModuleDict(layer_list)

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        for i in range(len(self.feature_list)-1):
            x = F.dropout(x, self.drop_rate, self.training)
            x = self.layer_list["conv"+str(i+1)](x, edge_index)
            if i < len(self.feature_list)-2:
                x = F.elu(x)
        return x

class MLP(torch.nn.Module):
    """ adapted from https://github.com/CUAI/CorrectAndSmooth/blob/master/gen_models.py """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout=.5):
        super(MLP, self).__init__()
        self.lins = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        if num_layers == 1:
            # just linear layer i.e. logistic regression
            self.lins.append(torch.nn.Linear(in_channels, out_channels))
        else:
            self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
            for _ in range(num_layers - 2):
                self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
                self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
            self.lins.append(torch.nn.Linear(hidden_channels, out_channels))


        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, data, input_tensor=False):
        if not input_tensor:
            x = data.x
        else:
            x = data
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = F.relu(x, inplace=True)
            x = self.bns[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x
    
class LINK(torch.nn.Module):
    """ logistic regression on adjacency matrix """
    
    def __init__(self, num_nodes, out_channels):
        super(LINK, self).__init__()
        self.W = torch.nn.Linear(num_nodes, out_channels)
        self.num_nodes = num_nodes

    def reset_parameters(self):
        self.W.reset_parameters()
        
    def forward(self, data):
        N = data.num_nodes
        edge_index = data.edge_index
        if isinstance(edge_index, torch.Tensor):
            row, col = edge_index
            row = row-row.min() # for sampling
            A = SparseTensor(row=row, col=col, sparse_sizes=(N, self.num_nodes)).to_torch_sparse_coo_tensor()
        elif isinstance(edge_index, SparseTensor):
            A = edge_index.to_torch_sparse_coo_tensor()
        logits = self.W(A)
        return logits

class LINK_Concat(torch.nn.Module):   
    """ concate A and X as joint embeddings i.e. MLP([A;X])"""

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, num_nodes, dropout=.5):  
        super(LINK_Concat, self).__init__() 
        self.mlp = MLP(in_channels + num_nodes, hidden_channels, out_channels, num_layers, dropout=dropout)
        self.in_channels = in_channels  
        self.x = None


    def reset_parameters(self): 
        self.mlp.reset_parameters() 

    def forward(self, data):    
        if not isinstance(self.x, torch.Tensor):
            N = data.x.shape[0]
            row, col = data.edge_index
            col = col + self.in_channels    
            feat_nz = data.x.nonzero(as_tuple=True)    
            feat_row, feat_col = feat_nz    
            full_row = torch.cat((feat_row, row))   
            full_col = torch.cat((feat_col, col))   
            value = data.x[feat_nz]    
            full_value = torch.cat((value,  
                                torch.ones(row.shape[0], device=value.device))) 
            x = SparseTensor(row=full_row, col=full_col,    
                         sparse_sizes=(N, N+self.in_channels)   
                            ).to_torch_sparse_coo_tensor()  
        else:
            x = self.x
        logits = self.mlp(x, input_tensor=True)
        return logits

class LINKX(torch.nn.Module):	
    """ our LINKX method with skip connections 
        a = MLP_1(A), x = MLP_2(X), MLP_3(sigma(W_1[a, x] + a + x))
    """

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, num_nodes, dropout, linkx_args):
        super(LINKX, self).__init__()	
        self.mlpA = MLP(num_nodes, hidden_channels, hidden_channels, linkx_args.link_init_layers_A, dropout=0)
        self.mlpX = MLP(in_channels, hidden_channels, hidden_channels, linkx_args.link_init_layers_X, dropout=0)
        self.W = torch.nn.Linear(2*hidden_channels, hidden_channels)
        self.mlp_final = MLP(hidden_channels, hidden_channels, out_channels, num_layers, dropout=dropout)
        self.in_channels = in_channels
        self.num_nodes = num_nodes
        self.A = None
        self.inner_activation = linkx_args.inner_activation
        self.inner_dropout = linkx_args.inner_dropout

    def reset_parameters(self):	
        self.mlpA.reset_parameters()	
        self.mlpX.reset_parameters()
        self.W.reset_parameters()
        self.mlp_final.reset_parameters()	

    def forward(self, data):	
        m = data.num_nodes	
        feat_dim = data.x
        row, col = data.edge_index
        row = row-row.min()
        A = SparseTensor(row=row, col=col,	
                 sparse_sizes=(m, self.num_nodes)
                        ).to_torch_sparse_coo_tensor()

        xA = self.mlpA(A, input_tensor=True)
        xX = self.mlpX(data.x, input_tensor=True)
        x = torch.cat((xA, xX), axis=-1)
        x = self.W(x)
        if self.inner_dropout:
            x = F.dropout(x)
        if self.inner_activation:
            x = F.relu(x)
        x = F.relu(x + xA + xX)
        x = self.mlp_final(x, input_tensor=True)

        return x


class H2GCNConv(torch.nn.Module):
    """ Neighborhood aggregation step """
    def __init__(self):
        super(H2GCNConv, self).__init__()

    def reset_parameters(self):
        pass

    def forward(self, x, adj_t, adj_t2):
        device = x.device
        adj_t = adj_t.to('cpu')
        adj_t2 = adj_t2.to('cpu')
        x = x.to('cpu')
        x1 = matmul(adj_t, x).to(device)
        x2 = matmul(adj_t2, x).to(device)
        return torch.cat([x1, x2], dim=1)

class H2GCN(torch.nn.Module):
    """ implementation from LINKX paper """
    def __init__(self, in_channels, hidden_channels, out_channels, edge_index, num_nodes,
                    num_layers=2, dropout=0.5, save_mem=False, num_mlp_layers=1,
                    use_bn=True, conv_dropout=True):
    #Note: DO NOT change the setting "num_layers=2"!!!!!
    #num_mlp_layers: num of mlp layers used in the first feature embedding stage
        
        super(H2GCN, self).__init__()

        self.feature_embed = MLP(in_channels, hidden_channels,
                hidden_channels, num_layers=num_mlp_layers, dropout=dropout)


        self.convs = torch.nn.ModuleList()
        self.convs.append(H2GCNConv())

        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels*2*len(self.convs) ) )

        for l in range(num_layers - 1):
            self.convs.append(H2GCNConv())
            if l != num_layers-2:       # Assume the num_layers=2
                self.bns.append(torch.nn.BatchNorm1d(hidden_channels*2*len(self.convs) ) )
            # if l != 0:         
            #     self.bns.append(nn.BatchNorm1d(hidden_channels*2*len(self.convs) ) )

        self.dropout = dropout
        self.activation = F.relu
        self.use_bn = use_bn
        self.conv_dropout = conv_dropout # dropout neighborhood aggregation steps

        self.jump = JumpingKnowledge('cat')
        last_dim = hidden_channels*(2**(num_layers+1)-1)
        self.final_project = torch.nn.Linear(last_dim, out_channels)

        self.num_nodes = num_nodes
        self.init_adj(edge_index)

    def reset_parameters(self):
        self.feature_embed.reset_parameters()
        self.final_project.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def init_adj(self, edge_index):
        """ cache normalized adjacency and normalized strict two-hop adjacency,
        neither has self loops
        """
        n = self.num_nodes
        
        if isinstance(edge_index, SparseTensor):
            dev = edge_index.device
            adj_t = edge_index
            adj_t = scipy.sparse.csr_matrix(adj_t.to_scipy())
            adj_t[adj_t > 0] = 1
            adj_t[adj_t < 0] = 0
            adj_t = SparseTensor.from_scipy(adj_t).to(dev)
        elif isinstance(edge_index, torch.Tensor):
            row, col = edge_index
            adj_t = SparseTensor(row=col, col=row, value=None, sparse_sizes=(n, n))

        adj_t.remove_diag(0)
        adj_t2 = matmul(adj_t, adj_t)
        adj_t2.remove_diag(0)
        adj_t = scipy.sparse.csr_matrix(adj_t.to_scipy())
        adj_t2 = scipy.sparse.csr_matrix(adj_t2.to_scipy())
        adj_t2 = adj_t2 - adj_t
        adj_t2[adj_t2 > 0] = 1
        adj_t2[adj_t2 < 0] = 0

        adj_t = SparseTensor.from_scipy(adj_t)
        adj_t2 = SparseTensor.from_scipy(adj_t2)
        
        adj_t = gcn_norm(adj_t, None, n, add_self_loops=False) 
        adj_t2 = gcn_norm(adj_t2, None, n, add_self_loops=False)

        self.adj_t = adj_t.to(edge_index.device)
        self.adj_t2 = adj_t2.to(edge_index.device)

    def forward(self, data):
        x = data.x
        n = data.x.shape[0]

        adj_t = self.adj_t
        adj_t2 = self.adj_t2
        
        x = self.feature_embed(data)
        x = self.activation(x)
        xs = [x]
        if self.conv_dropout:
            x = F.dropout(x, p=self.dropout, training=self.training)
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t, adj_t2) 
            if self.use_bn:
                x = self.bns[i](x)
            xs.append(x)
            if self.conv_dropout:
                x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t, adj_t2)
        if self.conv_dropout:
            x = F.dropout(x, p=self.dropout, training=self.training)
        xs.append(x)

        x = self.jump(xs)
        if not self.conv_dropout:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.final_project(x)
        return x

# class FALayer(nn.Module):
#     def __init__(self, g, in_dim, dropout):
#         super(FALayer, self).__init__()
#         self.g = g
#         self.dropout = nn.Dropout(dropout)
#         self.gate = nn.Linear(2 * in_dim, 1)
#         torch.nn.init.xavier_normal_(self.gate.weight, gain=1.414)

#     def edge_applying(self, edges):
#         h2 = torch.cat([edges.dst['h'], edges.src['h']], dim=1)
#         g = torch.tanh(self.gate(h2)).squeeze()
#         e = g * edges.dst['d'] * edges.src['d']
#         e = self.dropout(e)
#         return {'e': e, 'm': g}

#     def forward(self, h):
#         self.g.ndata['h'] = h
#         self.g.apply_edges(self.edge_applying)
#         self.g.update_all(fn.u_mul_e('h', 'e', '_'), fn.sum('_', 'z'))

#         return self.g.ndata['z']


# class FAGCN(nn.Module):
#     def __init__(self, g, in_dim, hidden_dim, out_dim, dropout, eps, layer_num=2):
#         super(FAGCN, self).__init__()
#         self.g = g
#         self.eps = eps
#         self.layer_num = layer_num
#         self.dropout = dropout

#         self.layers = nn.ModuleList()
#         for i in range(self.layer_num):
#             self.layers.append(FALayer(self.g, hidden_dim, dropout))

#         self.t1 = nn.Linear(in_dim, hidden_dim)
#         self.t2 = nn.Linear(hidden_dim, out_dim)
#         self.reset_parameters()

#     def reset_parameters(self):
#         torch.nn.init.xavier_normal_(self.t1.weight, gain=1.414)
#         torch.nn.init.xavier_normal_(self.t2.weight, gain=1.414)

#     def forward(self, h):
#         h = F.dropout(h, p=self.dropout, training=self.training)
#         h = torch.relu(self.t1(h))
#         h = F.dropout(h, p=self.dropout, training=self.training)
#         raw = h
#         for i in range(self.layer_num):
#             h = self.layers[i](h)
#             h = self.eps * raw + h
#         h = self.t2(h)
#         return F.log_softmax(h, 1)  
