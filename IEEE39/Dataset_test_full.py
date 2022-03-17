import torch
import torch_geometric
from torch_geometric.data import Data,Dataset
from torch_geometric.data import InMemoryDataset
import scipy.io
import numpy as np
from scipy.sparse import csr_matrix
from torch_geometric.loader import DataLoader
from networkx.convert_matrix import from_numpy_matrix


# Full dataset Bus features (P,Q,U,Theta); Edge features (Pact_in Qact_in X Rate_A)
mat = scipy.io.loadmat(
    'C:/Users/avarbella/Documents/MATLAB/test_cascades/GNN/T_p_test.mat')  # load .mat file of
# unfiltered contigencies and trasform it into a dictionary
of = scipy.io.loadmat(
     'C:/Users/avarbella/Documents/MATLAB/test_cascades/GNN/of_test.mat')
node_f = scipy.io.loadmat(
    'C:/Users/avarbella/Documents/MATLAB/test_cascades/GNN/Bf_test.mat')
edge_f = scipy.io.loadmat(
    'C:/Users/avarbella/Documents/MATLAB/test_cascades/GNN/Ef_test.mat')

T_p = mat['T_p_test']
of = of['of_test']
node_f = node_f['Bf_test']
edge_f = edge_f['Ef_test']
data_list = []

for i in range(len(T_p)):
    T_mask = np.ma.masked_values(T_p[i, 0], 0)
    T_sparse = csr_matrix(T_mask)
    D = torch_geometric.utils.from_scipy_sparse_matrix(T_sparse)
    G = from_numpy_matrix(T_p[i, 0], parallel_edges=False)

    x = torch.tensor(node_f[0,i],dtype=torch.float)
    f=torch.tensor(edge_f[i,0],dtype=torch.float)

    edge_index=D[0].clone().detach()
    D=Data(x=x,edge_attr=f,edge_index=edge_index, y = torch.tensor(of[i,0][0],dtype=int))
    D = D.pin_memory()
    D = D.to('cpu', non_blocking=True)
    data_list.append(D)

loader = DataLoader(data_list,shuffle=False)
torch.save(loader, 'C:/Users/avarbella/Documents/GraphGym-master/GraphGym-master/run/datasets/IEEE39/IEEE39/test_dataset.pt')

fl_pos=InMemoryDataset.collate(data_list)
torch.save(loader, 'C:/Users/avarbella/Documents/GraphGym-master/GraphGym-master/run/datasets/IEEE39/IEEE39/test_dataset1.pt')
# print("GraphData loader object composed of {} graphs, each graph has {} nodes, {} features/node, {} branches,"
#       "{} features/branch".format(len(loader),len(loader.dataset[0].x),
#             len(loader.dataset[0].x[1]),len(loader.dataset[0].edge_attr),len(loader.dataset[0].edge_attr[1])))
