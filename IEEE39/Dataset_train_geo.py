import torch
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
import scipy.io
import numpy as np
from scipy.sparse import csr_matrix
from torch_geometric.loader import DataLoader, DenseDataLoader
from torch.utils.data import WeightedRandomSampler
from deepsnap.graph import Graph
from deepsnap.dataset import GraphDataset
import networkx as nx
from networkx.convert_matrix import from_numpy_matrix
from torchsampler import ImbalancedDatasetSampler
import seaborn as sns
import matplotlib.pyplot as plt
import mat73

# Full dataset Bus features (P,Q,U,Theta); Edge feature (Pact_in/ Rate_A)
def get_class_distribution(dataset_obj):
    count_dict = {k: 0 for  k,v in enumerate(np.unique([dataset_obj[i].y for i in range(0,len(dataset_obj))]))}

    for i in range(len(dataset_obj)):
        y_lbl = dataset_obj[i].y.item()
        count_dict[y_lbl] += 1

    return count_dict


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

mat = mat73.loadmat(
    'C:/Users/avarbella/Documents/MATLAB/test_cascades/GNN/IEEE39_rnd_mix/DS_over_tot.mat')  # load .mat file of adjaciency matrices
of = mat73.loadmat(
     'C:/Users/avarbella/Documents/MATLAB/test_cascades/GNN/IEEE39_rnd_mix/label_over_tot.mat')
node_f = mat73.loadmat(
    'C:/Users/avarbella/Documents/MATLAB/test_cascades/GNN/IEEE39_rnd_mix/B_f_over_tot.mat')

#Data extract
T_p = mat['T_p_tot']
of = of['T_p_tot']
node_f = node_f['T_p_tot']
data_list = []

for i in range(len(T_p)):
    T_mask = np.ma.masked_values(T_p[i][0], 0)
    T_sparse = csr_matrix(T_mask)
    D = torch_geometric.utils.from_scipy_sparse_matrix(T_sparse)
    G = from_numpy_matrix(T_p[i][0], parallel_edges=False)
    # parallel_edges=False if entry of A is =!0 then is interpreted as edge weight not as number of edges
    x = torch.tensor(node_f[i][0], dtype=torch.float)
    edge_index = D[0].clone().detach()
    D = Data(x=x, edge_index=edge_index, y=torch.tensor(of[i][0].item(0), dtype=int)).to(device)
    data_list.append(D)

# class_count = [i for i in get_class_distribution(data_list).values()]
# class_weights = 1./torch.tensor(class_count, dtype=torch.float)
# weights =np.zeros(len(data_list))
# for i in range(len(data_list)):
#     if data_list[i].y.item() == 0:
#         weights[i] = (class_weights[0])
#     else:
#         weights[i] = (class_weights[1])
# sns.histplot(data=weights)
# weight=torch.tensor(weights,dtype=torch.float)
#print("Distribution of classes: \n", get_class_distribution(data_list))
#weighted_sampler =WeightedRandomSampler()
loader = DataLoader(data_list, batch_size=120,sampler=ImbalancedDatasetSampler(data_list))#WeightedRandomSampler(weights,len(data_list)))
torch.save(loader, 'C:/Users/avarbella/Documents/GraphGym-master/GraphGym-master/run/datasets/IEEE39/IEEE39/training_dataset_IEEE39mix.pt')
print("GraphData loader object composed of {} graphs, each graph has {} nodes, {} features/node".format(len(loader),len(loader.dataset[0].x),
            len(loader.dataset[0].x[1])))

plt.show()