import pandas as pd
import torch
import torch_geometric
from torch_geometric.data import Data
import numpy as np
from scipy.sparse import csr_matrix


data_list = []
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
edge_order = pd.read_csv('edge_oder118.csv', index_col=False, header=None)
edge_order = torch.tensor(edge_order.values)
edge_order_flip = torch.fliplr(edge_order)

# Full dataset Bus features (P,Q,U,Theta); Edge feature (Pact_in/ Rate_A)
device = torch.device("cuda:0")
A = pd.read_csv('C:\\Users\\avarbella\\Documents\\MATLAB\\test_cascades\\link_prediction_dataproc\\Adj.csv', header=None)
A = csr_matrix(np.asarray(A))
#A = coo_matrix((A.data,(A.i)))
X = pd.read_csv('C:\\Users\\avarbella\\Documents\\MATLAB\\test_cascades\\link_prediction_dataproc\\X_tot.csv', header=None)
X = X.astype('double')
B = pd.read_csv('C:\\Users\\avarbella\\Documents\\MATLAB\\test_cascades\\link_prediction_dataproc\\B_tot.csv', header=None)
B = B.astype('double')

X = torch.tensor(X.values, dtype=torch.float, device=device)
B = torch.tensor(B.values, dtype=torch.float)

T_sparse = csr_matrix(A)
D = torch_geometric.utils.from_scipy_sparse_matrix(T_sparse)
edge_index = D[0].clone().detach()

B_tot = torch.empty(size=(len(edge_index.H), 4000))
for j, val in enumerate(edge_order_flip):
    index = np.where((edge_index.H == edge_order[j]).all(axis=1))
    index_flip = np.where((edge_index.H == val).all(axis=1))
    B_tot[index]      = B[j]
    B_tot[index_flip] = B[j]
B_tot.to(device)
edge_index.to(device)
D = Data(x=X, edge_index=edge_index, edge_attr=B_tot, device=device).to(device)
torch.save(D, 'graph.pt')

