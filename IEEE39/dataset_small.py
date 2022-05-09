import pandas as pd
import torch
import torch_geometric
from torch_geometric.data import Data,Dataset
import numpy as np
from scipy.sparse import csr_matrix
from torch_geometric.loader import DataLoader,DenseDataLoader
from networkx.convert_matrix import from_numpy_matrix
import mat73
from torchsampler import ImbalancedDatasetSampler

data_list = []
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
edge_order = pd.read_csv('order_IEEE24_csv.csv', index_col=False, header=None)
edge_order = torch.tensor(edge_order.values)

# Full dataset Bus features (P,Q,U,Theta); Edge feature (Pact_in/ Rate_A)
mat = mat73.loadmat(
    'C:/Users/avarbella/Documents/MATLAB/test_cascades/GNN/UK/P_p_train.mat')  # load .mat file of
# unfiltered contigencies and trasform it into a dictionary
of = mat73.loadmat(
    'C:/Users/avarbella/Documents/MATLAB/test_cascades/GNN/UK/of_train.mat')
node_f = mat73.loadmat(
    'C:/Users/avarbella/Documents/MATLAB/test_cascades/GNN/UK/Bf_train.mat')
edge_f = mat73.loadmat(
    'C:/Users/avarbella/Documents/MATLAB/test_cascades/GNN/UK/Ef_train.mat')
T_p =      mat['P_p_train']
of =         of['of_train']
node_f = node_f['Bf_train']
edge_f = edge_f['Ef_train']

edge_order_flip = torch.fliplr(edge_order)
for i in range(len(T_p)):

    T_sparse = csr_matrix(T_p[i][0])
    D = torch_geometric.utils.from_scipy_sparse_matrix(T_sparse)

    x = torch.tensor(node_f[i][0], dtype=torch.float, device=device)
    f = torch.tensor(edge_f[i][0], dtype=torch.float)
    edge_index = D[0].clone().detach()
    f_tot = torch.empty(size=(len(edge_index.H), 4))
    for j, val in enumerate(edge_order_flip):
        index = np.where((edge_index.H == edge_order[j]).all(axis=1))
        index_flip = np.where((edge_index.H == val).all(axis=1))
        f_tot[index] = f[j]
        f_tot[index_flip] = f[j]
    f_tot.to(device)
    edge_index.to(device)
    D = Data(x=x, edge_index=edge_index, edge_attr=f_tot, y=torch.tensor(of[i][0].item(0), dtype=int, device=device)).to(device)
    data_list.append(D)

loader = DataLoader(data_list, batch_size=128, shuffle=True)
torch.save(loader, 'IEEE24_train.pt')

print("GraphData loader object composed of {} graphs, each graph has {} nodes, {} features/node"
      .format(len(loader), len(loader.dataset[0].x), len(loader.dataset[0].x[1])))