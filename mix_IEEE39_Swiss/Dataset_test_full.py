import torch
import torch_geometric
from torch_geometric.data import Data,Dataset
from torch_geometric.data import InMemoryDataset
import scipy.io
import numpy as np
from scipy.sparse import csr_matrix
from torch_geometric.loader import DataLoader
from networkx.convert_matrix import from_numpy_matrix
import mat73
#from torchsampler import ImbalancedDatasetSampler
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Full dataset Bus features (P,Q,U,Theta); Edge feature (Pact_in/ Rate_A)
mat = mat73.loadmat(
    'C:/Users/avarbella/Documents/MATLAB/test_cascades/GNN/mix_39_118_swiss/half_dataset/T_p_test_tot.mat')  # load .mat file of
# unfiltered contigencies and trasform it into a dictionary
of = mat73.loadmat(
     'C:/Users/avarbella/Documents/MATLAB/test_cascades/GNN/mix_39_118_swiss/half_dataset/of_test_tot.mat')
node_f = mat73.loadmat(
    'C:/Users/avarbella/Documents/MATLAB/test_cascades/GNN/mix_39_118_swiss/half_dataset/Bf_test_tot.mat')
T_p = mat['y']
of = of['y']
node_f = node_f['y']
data_list = []

#print([of[i][0][0] for i in range(len(of))])
for i in range(len(T_p)):
    T_mask = np.ma.masked_values(T_p[i][0], 0)
    T_sparse = csr_matrix(T_mask)
    D = torch_geometric.utils.from_scipy_sparse_matrix(T_sparse)
    G = from_numpy_matrix(T_p[i][0], parallel_edges=False)

    x = torch.tensor(node_f[i][0], dtype=torch.float,device=device)

    edge_index = D[0].clone().detach().to(device)
    D = Data(x=x, edge_index=edge_index, y=torch.tensor(of[i][0].item(0), dtype=int,device=device)).to(device)
    #D = D.pin_memory()
   # D = D.to('cpu', non_blocking=True)
    data_list.append(D)

loader = DataLoader(data_list, batch_size=120,shuffle=True)
torch.save(loader, 'C:/Users/avarbella/Documents/GraphGym-master/GraphGym-master/run/datasets/IEEE39/IEEE39/test_118_swiss_small.pt')

print("GraphData loader object composed of {} graphs, each graph has {} nodes, {} features/node"
      .format(len(loader), len(loader.dataset[0].x),
            len(loader.dataset[0].x[1])))
