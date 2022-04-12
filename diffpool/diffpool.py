import os
from math import ceil
import torch
import logging
import torch.nn as nn
import torch.nn.functional as F
from graphgym.utils.agg_runs import agg_runs
from torch_geometric.nn import DenseSAGEConv, dense_mincut_pool,DenseGraphConv, DenseGCNConv,global_max_pool
from torch_geometric.utils import to_dense_adj, to_dense_batch
from graphgym.logger import setup_printing, create_logger
import sklearn


def is_eval_epoch(cur_epoch, eval_period,max_epoch):
    """Determines if the model should be evaluated at the current epoch."""
    return (
            (cur_epoch + 1) % eval_period == 0 or
            cur_epoch == 0 or
            (cur_epoch + 1) == max_epoch
    )

eval_period = 5
max_epoch = 100
dataset = torch.load('C:/Users/avarbella/Documents/GraphGym-master/'
                     'GraphGym-master/run/datasets/IEEE39/IEEE39/train_IEEE118_swiss_small_geo.pt')
data_test = torch.load('C:/Users/avarbella/Documents/GraphGym-master/'
                       'GraphGym-master/run/datasets/IEEE39/IEEE39/test_IEEE118_swiss_small_geo.pt')

train_dataset = dataset
val_loader = data_test
train_loader = dataset
datasets = [train_loader, val_loader]


class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 normalize=True, lin=True):
        super().__init__()

        self.conv1 = DenseGraphConv(in_channels, hidden_channels, aggr='max')
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv2 = DenseGraphConv(hidden_channels, hidden_channels, aggr='max')
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv3 = DenseGraphConv(hidden_channels, out_channels, aggr='max')
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

        x = torch.cat([x1, x3], dim=-1)

        if self.lin is not None:
            x = F.relu(self.lin(x))

        return x


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        n_neurons = 32
        num_nodes = ceil(0.25 * 100)
        self.gnn1_pool = GNN(dataset.dataset[0].num_features, n_neurons, num_nodes)
        self.gnn1_embed = GNN(dataset.dataset[0].num_features, n_neurons, n_neurons, lin=False)

        num_nodes = ceil(0.25 * num_nodes)
        self.gnn2_pool = GNN(2 * n_neurons, n_neurons, num_nodes)
        self.gnn2_embed = GNN(2 * n_neurons, n_neurons, n_neurons, lin=False)

        self.gnn3_embed = GNN(2 * n_neurons, n_neurons, n_neurons, lin=False)

        #self.lin1 = torch.nn.Linear(2 * n_neurons, n_neurons)
        #
        #self.lin2 = torch.nn.Linear(n_neurons, n_neurons)
        #self.lin3 = torch.nn.Linear(n_neurons, 2)
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2 * n_neurons, n_neurons),
            nn.Dropout(p=0.1),
            nn.ReLU(),
            nn.Linear(n_neurons, n_neurons),
            nn.Dropout(p=0.1),
            nn.ReLU(),
            nn.Linear(n_neurons, 2)
        )
    def init_xavier(self):
        torch.manual_seed(0)

        def init_weights(m):
            if type(m) == nn.Linear and m.weight.requires_grad and m.bias.requires_grad:
                if self.activation_name in ['tanh', 'relu']:
                    gain = nn.init.calculate_gain(self.activation_name)
                else:
                    gain = 1
                torch.nn.init.xavier_uniform_(m.weight, gain=gain)
                m.bias.data.fill_(0)

        self.apply(init_weights)
    def forward(self, x, adj, mask=None):
        s = self.gnn1_pool(x, adj, mask)
        x = self.gnn1_embed(x, adj, mask)

        x, adj, l1, e1 = dense_mincut_pool(x, adj, s, mask)

        s = self.gnn2_pool(x, adj)
        x = self.gnn2_embed(x, adj)
        #
        x, adj, l2, e2 = dense_mincut_pool(x, adj, s)

        x = self.gnn3_embed(x, adj)

        x,_ = x.max(dim=1)
        #x1=x.mean(dim=1)
        #x = F.relu(self.lin1(x))
        #x = F.relu(self.lin2(x))
        #x = self.lin3(x)
        mlp = self.layers(x)
        return F.log_softmax(mlp, dim=-1) #, l1 + l2, e1 + e2


device = 'cuda'
#torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001, weight_decay=1e-4)


def train_epoch(logger, loader, model, optimizer):
    model.train()
    for data in loader:
        optimizer.zero_grad()
        data.to(torch.device(device))
        adj = to_dense_adj(data.edge_index, batch=data.batch, edge_attr=data.edge_weight)
        data1, _ = to_dense_batch(data.x, batch=data.batch)
        output = model(data1, adj)
        loss = F.nll_loss(output, data.y.view(-1))
        pred = model(data1, adj).max(dim=1)[1]
        loss.backward()
        optimizer.step()
        logger.update_stats(true=data.y.view(-1).detach().cpu(),
                            pred=pred.detach().cpu(),
                            loss=loss.item(),
                            lr=0.0001, time_used=0, params=0)

#@torch.no_grad()
def eval_epoch(logger, loader, model):
    model.eval()
    for data in loader:
        data.to(torch.device(device))
        adj = to_dense_adj(data.edge_index, batch=data.batch, edge_attr=data.edge_weight)
        data1, _ = to_dense_batch(data.x, batch=data.batch)
        x = data.y.view(-1).type(torch.DoubleTensor)
        pred = model(data1, adj).max(dim=1)[1]
        loss = 0
        logger.update_stats(true=data.y.view(-1).detach().cpu(),
                            pred=pred.detach().cpu(),
                            loss=loss,
                            lr=0, time_used=0,params=0)


def train(loggers, loaders, model, optimizer):
    start_epoch = 0
    logging.info('Start from epoch {}'.format(start_epoch))
    num_splits = len(loggers)
    for cur_epoch in range(start_epoch, max_epoch):
        train_epoch(loggers[0], loaders[0], model, optimizer)
        loggers[0].write_epoch(cur_epoch)

        if is_eval_epoch(cur_epoch, eval_period, max_epoch):
            for i in range(1, num_splits):
                eval_epoch(loggers[i], loaders[i], model)
                loggers[i].write_epoch(cur_epoch)
    for logger in loggers:
        logger.close()
    logging.info('Task done, results saved in {}'.format('C:/Users/avarbella/Documents/GraphGym-master/GraphGym-master/diffpool/results2/'))

setup_printing()
meters = create_logger(datasets)
loaders = datasets
train(meters, loaders, model, optimizer)
#os.path.abspath(os.path.join(yourpath, os.pardir))
#agg_runs('C:/Users/avarbella/Documents/GraphGym-master/GraphGym-master/diffpool/results2/', 'auto')
torch.save(model.state_dict(),'C:/GNN/modelGAT.pth')
#torch.save(model.state_dict(), os.get_parent_dir('results2', "modelGAT.pth"))