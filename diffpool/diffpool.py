import os
from math import ceil
import torch
import logging
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn import DenseSAGEConv, dense_diff_pool,DenseGraphConv, DenseGCNConv
from torch_sparse import SparseTensor
from sklearn.metrics import balanced_accuracy_score, f1_score
from torch_geometric.utils import to_dense_adj, to_dense_batch
from graphgym.logger import setup_printing, create_logger


def is_eval_epoch(cur_epoch, eval_period,max_epoch):
    """Determines if the model should be evaluated at the current epoch."""
    return (
            (cur_epoch + 1) % eval_period == 0 or
            cur_epoch == 0 or
            (cur_epoch + 1) == max_epoch
    )

eval_period = 5
max_epoch = 200
dataset = torch.load('C:/Users/avarbella/Documents/GraphGym-master/'
                     'GraphGym-master/run/datasets/IEEE39/IEEE39/training_dataset_IEEE39mix.pt')
data_test = torch.load('C:/Users/avarbella/Documents/GraphGym-master/'
                       'GraphGym-master/run/datasets/IEEE39/IEEE39/test_dataset_Aw.pt')

train_dataset = dataset
val_loader = data_test
train_loader = dataset
datasets =[train_loader, val_loader]


class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 normalize=False, lin=True):
        super().__init__()

        self.conv1 = DenseGraphConv(in_channels, hidden_channels, aggr='mean')
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv2 = DenseGraphConv(hidden_channels, hidden_channels, aggr='mean')
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv3 = DenseGraphConv(hidden_channels, out_channels, aggr='mean')
        self.bn3 = torch.nn.BatchNorm1d(out_channels)

        if lin is True:
            self.lin = torch.nn.Linear( hidden_channels + out_channels,
                                       out_channels)
        else:
            self.lin = None

    def bn(self, i, x):
        batch_size, num_nodes, num_channels = x.size()

        x = x.view(-1, num_channels)
        x = getattr(self, f'bn{i}')(x)
        x = x.view(batch_size, num_nodes, num_channels)
        return x

    def forward(self, x, adj, mask=None):
        batch_size,num_nodes, in_channels = x.size()

        x0 = x
        x1 = self.bn(1, F.relu(self.conv1(x0, adj, mask)))
        #x2 = self.bn(2, F.relu(self.conv2(x1, adj, mask)))
        x3 = self.bn(3, F.relu(self.conv3(x1, adj, mask)))

        x = torch.cat([x1,  x3], dim=-1)

        if self.lin is not None:
            x = F.relu(self.lin(x))

        return x


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()

        num_nodes = ceil(0.5 * 100)
        self.gnn1_pool = GNN(dataset.dataset[0].num_features, 64, num_nodes)
        self.gnn1_embed = GNN(dataset.dataset[0].num_features, 64, 64, lin=False)

        num_nodes = ceil(0.5 * num_nodes)
        self.gnn2_pool = GNN(2 * 64, 64, num_nodes)
        self.gnn2_embed = GNN(2 * 64, 64, 64, lin=False)

        self.gnn3_embed = GNN(2 * 64, 64, 64, lin=False)

        self.lin1 = torch.nn.Linear(2 * 64, 512)
        self.lin2 = torch.nn.Linear(512, 512)
        self.lin3 = torch.nn.Linear(512, 2)

    def forward(self, x, adj, mask=None):
        s = self.gnn1_pool(x, adj, mask)
        x = self.gnn1_embed(x, adj, mask)

        x, adj, l1, e1 = dense_diff_pool(x, adj, s, mask)

        s = self.gnn2_pool(x, adj)
        x = self.gnn2_embed(x, adj)

        x, adj, l2, e2 = dense_diff_pool(x, adj, s)

        x = self.gnn3_embed(x, adj)

        x = x.mean(dim=1)
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = self.lin3(x)
        return F.log_softmax(x, dim=-1), l1 + l2, e1 + e2


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.00005)


def train_epoch(logger, loader, model, optimizer):
    model.train()
    for data in loader:
        optimizer.zero_grad()
        data.to(torch.device(device))
        adj = to_dense_adj(data.edge_index, batch=data.batch, edge_attr=data.edge_weight)
        data1, _ = to_dense_batch(data.x, batch=data.batch)
        output, _, _ = model(data1, adj)
        loss = F.nll_loss(output, data.y.view(-1))
        pred = model(data1, adj)[0].max(dim=1)[1]
        loss.backward()
        optimizer.step()
        logger.update_stats(true=data.y.view(-1).detach().cpu(),
                            pred=pred.detach().cpu(),
                            loss=loss.item(),
                            lr=0.0001)

#@torch.no_grad()
def eval_epoch(logger, loader, model):
    model.eval()
    for data in loader:
        data.to(torch.device(device))
        adj = to_dense_adj(data.edge_index, batch=data.batch, edge_attr=data.edge_weight)
        data1, _ = to_dense_batch(data.x, batch=data.batch)
        x=data.y.view(-1).type(torch.DoubleTensor)
        pred = model(data1, adj)[0].max(dim=1)[1].type(torch.DoubleTensor)
        #pred = model(data1, adj)#[0].max(dim=1)[1]
        loss = 0
        #pred = model(data1, adj)[0].max(dim=1)[1]
        logger.update_stats(true=data.y.view(-1).detach().cpu(),
                            pred=pred.detach().cpu(),
                            loss=loss,
                            lr=0)


def train(loggers, loaders, model, optimizer):
    start_epoch = 0

    logging.info('Start from epoch {}'.format(start_epoch))

    num_splits = len(loggers)
    for cur_epoch in range(start_epoch, max_epoch):
        train_epoch(loggers[0], loaders[0], model, optimizer)
        loggers[0].write_epoch(cur_epoch)

        if is_eval_epoch(cur_epoch,eval_period,max_epoch):
            for i in range(1, num_splits):
                eval_epoch(loggers[i], loaders[i], model)
                loggers[i].write_epoch(cur_epoch)
    for logger in loggers:
        logger.close()
    logging.info('Task done, results saved in {}'.format(os.chdir("results2")))

setup_printing()
meters = create_logger(datasets)
loaders = datasets
train(meters, loaders, model, optimizer)
#def train(epoch):
#    model.train()
#    loss_all = 0
#
#    for data in train_loader:
#        data = data.to(device)
#        optimizer.zero_grad()
#        adj = to_dense_adj(data.edge_index, batch=data.batch, edge_attr=data.edge_weight)
#        data1, _ = to_dense_batch(data.x, batch=data.batch)
#        output, _, _ = model(data1, adj)
#        loss = F.nll_loss(output, data.y.view(-1))
#        loss.backward()
#        loss_all += data.y.size(0) * loss.item()
#        optimizer.step()
#    return loss_all / len(train_dataset)



#def test(loader):
#    model.eval()
#    correct = 0
#    pred_list = []
#    correct_list = []
#    for data in loader:
#        data = data.to(device)
#        adj = to_dense_adj(data.edge_index, batch=data.batch, edge_attr=data.edge_weight)
#        data1, _ = to_dense_batch(data.x, batch=data.batch)
#        pred = model(data1, adj)[0].max(dim=1)[1]
#        pred_list.append(pred.cpu().detach().numpy())
#        correct += pred.eq(data.y.view(-1)).sum().item()
#        correct_list.append(data.y.view(-1).cpu().detach().numpy())
#
#        f1 = f1_score(data.y.view(-1).cpu().detach().numpy(), pred.cpu().detach().numpy())
#        bal_acc = balanced_accuracy_score(data.y.view(-1).cpu().detach().numpy(), pred.cpu().detach().numpy())
#    return correct / len(loader.dataset), f1, bal_acc


#best_val_acc = test_acc = 0
#for epoch in range(1, 151):
#    train_loss = train(epoch)
#    val_acc, val_f1, val_balacc = test(val_loader)
#    train_acc, train_f1, train_balacc = test(train_loader)
#
#    if val_acc > best_val_acc:
#        best_val_acc = val_acc
#    print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, '
#          f'Train acc: {train_acc:.4f}, Train f1: {train_f1:.4f}, Train bal acc: {train_balacc:.4f}, '
#          f'Val Acc: {val_acc:.4f}, Val f1: {val_f1:.4f}, Val bal acc: {val_balacc:.4f}')
