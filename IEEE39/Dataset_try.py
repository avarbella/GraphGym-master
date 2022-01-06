import torch
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
import scipy.io
import numpy as np
from scipy.sparse import csr_matrix
from torch_geometric.loader import DataLoader
from deepsnap.graph import Graph
from deepsnap.dataset import GraphDataset
import networkx as nx
from networkx.convert_matrix import from_numpy_matrix


mat = scipy.io.loadmat(
    'C:/Users/avarbella/Documents/MATLAB/test_cascades/GNN/T_p.mat')  # load .mat file of
# unfiltered contigencies and trasform it into a dictionary
of =  scipy.io.loadmat(
     'C:/Users/avarbella/Documents/MATLAB/test_cascades/GNN/of.mat')
node_f = scipy.io.loadmat(
    'C:/Users/avarbella/Documents/MATLAB/test_cascades/GNN/node_f.mat')
T_p = mat['T_p']
of = of['output_features']
node_f = node_f['B_f']

data_list=[]
cont=[]
counter =0
h=0
for i in range(len(T_p)):
    T_mask = np.ma.masked_values(T_p[i, 0], 0)
    T_sparse = csr_matrix(T_mask)
    D = torch_geometric.utils.from_scipy_sparse_matrix(T_sparse)
    G = from_numpy_matrix(T_p[i, 0], parallel_edges=False)

    for j in range(len(G.nodes())):
        # print((node_f[0, h][j][0]))
        G.nodes[j]['P'] = node_f[0, h][j][0]
        G.nodes[j]['V'] = node_f[0, h][j][2]
    x = torch.tensor(list(nx.get_node_attributes(G, 'P').values()), dtype=torch.int)
    x = torch.reshape(x, (len(G.nodes()), 1))
    w = torch.tensor(list(nx.get_node_attributes(G, 'V').values()), dtype=torch.int)
    w = torch.reshape(w, (len(G.nodes()), 1))
    f = torch.cat((x, w), 1)

    edge_index=D[0].clone().detach()
    D=Data(x=f, edge_index=edge_index, y = torch.tensor(of[i,0][0],dtype=int))
    data_list.append(D)
    cont.append(data_list[i].num_edges)

fl_pos=InMemoryDataset.collate(data_list)
torch.save(fl_pos, 'C:/Users/avarbella/Documents/GraphGym-master/GraphGym-master/run/datasets/IEEE39/IEEE39/data.pt')
#fl_pos=data_list.Dataset(root= 'C:/Users/avarbella/Documents/GraphGym-master/GraphGym-master/run/datasets/IEEE39/')
#fl_pos.collate(data_list)
loader = DataLoader(data_list,batch_size=32)


#print(data.is_directed())
print(D.num_features)
# print(data.num_nodes)
#('C:/Users/avarbella/Documents/GraphGym-master/GraphGym-master/run/datasets/IEEE39/')