import os.path as osp
from math import ceil
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn import DenseSAGEConv, dense_diff_pool, DenseGCNConv,GCNConv
from torch_sparse import SparseTensor
from sklearn.metrics import balanced_accuracy_score, f1_score
from torch_geometric.utils import to_dense_adj, to_dense_batch
import warnings


warnings.filterwarnings("ignore", category=UserWarning)

dataset = torch.load('C:/Users/avarbella/Documents/GraphGym-master/'
                     'GraphGym-master/run/datasets/IEEE39/IEEE39/training_dataset_Aw.pt') #load train dataset
data_test = torch.load('C:/Users/avarbella/Documents/GraphGym-master/' 
                       'GraphGym-master/run/datasets/IEEE39/IEEE39/test_dataset_Aw.pt') #load test dataset

train_dataset = dataset
val_loader = data_test
train_loader = dataset
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DenseGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 normalize=False, lin=True):
        super().__init__()

        self.conv1 = DenseGCNConv(in_channels, hidden_channels)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv2 = DenseGCNConv(hidden_channels, hidden_channels)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv3 = DenseGCNConv(hidden_channels, out_channels)
        self.bn3 = torch.nn.BatchNorm1d(out_channels)

        if lin is True:
            self.lin = torch.nn.Linear(2*hidden_channels + out_channels,
                                       out_channels)
        else:
            self.lin = None

    def bn(self, i, x, f):
        if f == 2:
            num_nodes, num_channels = x.size()
        else:
            batch, num_nodes, num_channels = x.size()

        x = x.view(-1, num_channels)
        x = getattr(self, f'bn{i}')(x)
        x = x.view( num_nodes, num_channels)
        return x

    def forward(self, x, adj, mask=None, f=None):
     #   batch, num_nodes, in_channels = x.size()

        x0 = x
        x1 = self.bn(1, F.relu(self.conv1(x0, adj)),f)
        x2 = self.bn(2, F.relu(self.conv2(x1, adj)),f)
        x3 = self.bn(3, F.relu(self.conv3(x2, adj)),f)

        x = torch.cat([x1, x2, x3], dim=-1)

        if self.lin is not None:
            x = F.relu(self.lin(x))
        return x


class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 normalize=False, lin=True):
        super().__init__()

        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv3 = GCNConv(hidden_channels, out_channels)
        self.bn3 = torch.nn.BatchNorm1d(out_channels)

        if lin is True:
            self.lin = torch.nn.Linear(2*hidden_channels + out_channels,
                                       out_channels)
        else:
            self.lin = None

    def bn(self, i, x, f):
        if f == 2:
            num_nodes, num_channels = x.size()
        else:
            batch, num_nodes, num_channels = x.size()

        x = x.view(-1, num_channels)
        x = getattr(self, f'bn{i}')(x)
        x = x.view(num_nodes, num_channels)
        return x

    def forward(self, x, edge_index, adj, mask=None, f=None):
     #   batch, num_nodes, in_channels = x.size()

        x0 = x
        x1 = self.bn(1, F.relu(self.conv1(x0, adj)),f)
        x2 = self.bn(2, F.relu(self.conv2(x1, adj)),f)
        x3 = self.bn(3, F.relu(self.conv3(x2, adj)),f)

        x = torch.cat([x1, x2, x3], dim=-1)

        if self.lin is not None:
            x = F.relu(self.lin(x))
        return x


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()

        num_nodes = ceil(0.75 * 39)
        self.gnn1_pool = DenseGNN(dataset.dataset[0].num_features, 64, num_nodes)
        self.gnn1_embed = DenseGNN(dataset.dataset[0].num_features, 64, 64, lin=False)

        num_nodes = ceil(0.75 * num_nodes)
        self.gnn2_pool = DenseGNN(3 * 64, 64, num_nodes)
        self.gnn2_embed = DenseGNN(3 * 64, 64, 64, lin=False)

        self.gnn3_embed = DenseGNN(3 * 64, 64, 64, lin=False)

        self.lin1 = torch.nn.Linear(3 * 64, 64)
        self.lin2 = torch.nn.Linear(64, 1)

    def forward(self, x,  adj, mask=None, ):
        s = self.gnn1_pool(x, adj, mask, f=1)
        x1 = self.gnn1_embed(x, adj, mask, f=1)

        x, adj, l1, e1 = dense_diff_pool(x1, adj, s)
        s = self.gnn2_pool(x, adj, mask, f=1)
        x = self.gnn2_embed(x, adj, mask, f=1)

        x, adj, l2, e2 = dense_diff_pool(x, adj, s)

        x = self.gnn3_embed(x, adj, mask,f=1)
        x = x.mean(dim=1)

        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1), l1 + l2, e1 + e2


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def train(epoch):
    model.train()
    loss_all = 0

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        adj = to_dense_adj(data.edge_index, batch = data.batch, edge_attr=data.edge_weight)
        print(adj.dim)
        data, _ = to_dense_batch(data.x, batch=data.batch)
        output, _, _ = model(data, adj)
        loss = F.nll_loss(output, data.y.view(-1))
        loss.backward()
        loss_all += data.y.size(0) * loss.item()
        optimizer.step()
    return loss_all / len(train_dataset)


@torch.no_grad()
def test(loader):
    model.eval()
    correct = 0
    pred_list = []
    correct_list = []
    for data in loader:
        data = data.to(device)
        adj = to_dense_adj(data.edge_index, batch= data.batch, edge_attr=data.edge_weight)
        pred = model(data.x, data.edge_index, adj)[0].max(dim=1)[1]
        pred_list.append(pred.cpu().detach().numpy())
        correct_list.append(data.y.view(-1).cpu().detach().numpy())
        correct += pred.eq(data.y.view(-1)).sum().item()

    f1 = f1_score(correct_list, pred_list, zero_division=0)
    bal_acc = balanced_accuracy_score(correct_list, pred_list)
    return correct / len(loader.dataset), f1, bal_acc


best_val_acc = test_acc = 0
for epoch in range(100):
    train_loss = train(epoch)
    val_acc, val_f1, val_balacc = test(val_loader)
    train_acc, train_f1, train_balacc = test(train_loader)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
    print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, '
          f'Train acc: {train_acc:.4f}, Train f1: {train_f1:.4f}, Train bal acc: {train_balacc:.4f}, '
          f'Val Acc: {val_acc:.4f}, Val f1: {val_f1:.4f}, Val bal acc: {val_balacc:.4f}')
