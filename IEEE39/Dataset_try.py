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
    'C:/Users/avarbella/Documents/MATLAB/test_cascades/GNN/DS_over.mat')  # load .mat file of
# unfiltered contigencies and trasform it into a dictionary
of =  scipy.io.loadmat(
     'C:/Users/avarbella/Documents/MATLAB/test_cascades/GNN/label_over.mat')
node_f = scipy.io.loadmat(
    'C:/Users/avarbella/Documents/MATLAB/test_cascades/GNN/node_f.mat')
edge_f = scipy.io.loadmat(
    'C:/Users/avarbella/Documents/MATLAB/test_cascades/GNN/E_f.mat')
T_p = mat['DS_over']
of = of['label_over']
node_f = node_f['B_f']
edge_f = edge_f['E_f_post']

data_list=[]

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
        G.nodes[j]['Q'] = node_f[0, h][j][1]
        G.nodes[j]['S'] = node_f[0, h][j][2]
        G.nodes[j]['V'] = node_f[0, h][j][3]
        G.nodes[j]['A'] = node_f[0, h][j][4]
    x = torch.tensor(list(nx.get_node_attributes(G, 'P').values()), dtype=torch.int)
    x = torch.reshape(x, (len(G.nodes()), 1))
    w = torch.tensor(list(nx.get_node_attributes(G, 'V').values()), dtype=torch.int)
    w = torch.reshape(w, (len(G.nodes()), 1))
    w1 = torch.tensor(list(nx.get_node_attributes(G, 'Q').values()), dtype=torch.int)
    w1 = torch.reshape(w, (len(G.nodes()), 1))
    #w2 = torch.tensor(list(nx.get_node_attributes(G, 'S').values()), dtype=torch.int)
    #w2 = torch.reshape(w, (len(G.nodes()), 1))
    w3 = torch.tensor(list(nx.get_node_attributes(G, 'A').values()), dtype=torch.int)
    w3 = torch.reshape(w, (len(G.nodes()), 1))
    x = torch.cat((x, w, w1, w3), 1)
    f=torch.tensor(edge_f[i,0],dtype=torch.int)


    edge_index=D[0].clone().detach()
    D=Data(x=x, edge_attr=f,edge_index=edge_index, y = torch.tensor(of[i,0][0],dtype=int))
    data_list.append(D)
    #cont.append(data_list[i].num_edges)
    counter += 1

    if counter >= (h + 1) * 150:
        h += 1
fl_pos=InMemoryDataset.collate(data_list)

torch.save(fl_pos, 'C:/Users/avarbella/Documents/GraphGym-master/GraphGym-master/run/datasets/IEEE39/IEEE39/data.pt')
#fl_pos=data_list.Dataset(root= 'C:/Users/avarbella/Documents/GraphGym-master/GraphGym-master/run/datasets/IEEE39/')
#fl_pos.collate(data_list)
loader = DataLoader(data_list)
dp=GraphDataset.pyg_to_graphs(loader)
torch.save(loader, 'C:/Users/avarbella/Documents/GraphGym-master/GraphGym-master/run/datasets/IEEE39/IEEE39/data1.pt')


#print(data.is_directed())
print(D)
# print(data.num_nodes)
#('C:/Users/avarbella/Documents/GraphGym-master/GraphGym-master/run/datasets/IEEE39/')