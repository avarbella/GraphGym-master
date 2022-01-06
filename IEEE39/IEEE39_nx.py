import networkx as nx
import torch
import torch_geometric
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from networkx.convert_matrix import from_numpy_matrix
import scipy.io
import numpy as np
from scipy.sparse import csr_matrix
from deepsnap.graph import Graph
from deepsnap.dataset import GraphDataset
from torch_geometric.loader import DataLoader


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
    G = from_numpy_matrix(T_p[i,0], parallel_edges=False)
    #nx.set_node_attributes(G,node_f[0, h][0,:][0],'net_P')

    for j in range(len(G.nodes())):
         #print((node_f[0, h][j][0]))
        G.nodes[j]['P'] = node_f[0, h][j][0]
        G.nodes[j]['V'] = node_f[0, h][j][2]
    x=torch.tensor(list(nx.get_node_attributes(G,'P').values()),dtype=torch.int)
    x=torch.reshape(x,(len(G.nodes()),1))
    w=torch.tensor(list(nx.get_node_attributes(G,'V').values()),dtype=torch.int)
    w=torch.reshape(w,(len(G.nodes()),1))
    f=torch.cat((x,w),1)
    #torch.reshape(x,(len(G.nodes()),1))
    Graph.add_node_attr(G,'node_feature',f )

    DN = Graph(G,graph_label=of[i,0][0])

    data_list.append(DN)
    counter += 1

    if counter >= (h+1)*150:
        h += 1

    # D = torch_geometric.utils.from_scipy_sparse_matrix(T_sparse)
    # edge_index=torch.tensor(D[0],dtype=torch.long)
    # data_list.append(Data(x=None, edge_index=edge_index, y = of[i,0][0][0]))
    # graph = Graph.pyg_to_graph(data_list[i])

data = GraphDataset(data_list, task='graph', netlib= 'nx', minimum_node_per_graph=0)
#torch.save(data,'C:/Users/avarbella/Documents/GraphGym-master/'
               # 'GraphGym-master/run/datasets/IEEE39.gpickle')
nx.write_gpickle(data,'C:/Users/avarbella/Documents/GraphGym-master/'
                'GraphGym-master/run/datasets/IEEE39.gpickle')
#nx.draw_networkx(G,with_labels=True)

#print(x)
print(DN.num_graph_labels)
print(x)
print(DN.num_node_features)
#print(DN.num_edges)
#plt.show()
