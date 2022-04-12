import os
from math import ceil
import torch
import logging
import torch.nn as nn
import torch.nn.functional as F
from graphgym.utils.agg_runs import agg_runs
from torch_geometric.nn import GCNConv,global_max_pool, GATConv,GINConv, global_mean_pool, global_sort_pool,global_add_pool
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
max_epoch = 400
dataset = torch.load('C:/Users/avarbella/Documents/GraphGym-master/'
                     'GraphGym-master/run/datasets/IEEE39/IEEE39/train_full_swiss_small_geo.pt')
data_test = torch.load('C:/Users/avarbella/Documents/GraphGym-master/'
                       'GraphGym-master/run/datasets/IEEE39/IEEE39/test_full_swiss_small_geo.pt')

train_dataset = dataset
val_loader = data_test
train_loader = dataset
datasets = [train_loader, val_loader]


class Net(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        n_att_heads = 1
        self.conv1 = GATConv(in_channels, hidden_channels, edge_dim=1, heads=n_att_heads)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels * n_att_heads)
        compress_channels = 4
        self.conv2 = GATConv(n_att_heads * hidden_channels+in_channels, n_att_heads * hidden_channels+in_channels, edge_dim=1)
        self.bn2 = torch.nn.BatchNorm1d(n_att_heads * hidden_channels+in_channels)
        self.conv3 = GATConv(n_att_heads * hidden_channels+in_channels, n_att_heads * hidden_channels+in_channels, edge_dim=1)
        self.bn3 = torch.nn.BatchNorm1d(n_att_heads * hidden_channels+in_channels)
        self.layers = nn.Sequential(
            #nn.Flatten(),
            nn.Linear((n_att_heads * hidden_channels+in_channels) * 3, (n_att_heads * hidden_channels+in_channels) * 3),
            nn.Dropout(p=0.0),
            nn.PReLU(),
            #nn.Linear(compress_channels * 3, compress_channels * 3),
            #nn.Dropout(p=0.0),
            #nn.PReLU(),
            #nn.Linear(compress_channels * 3, compress_channels * 3),
            #nn.Dropout(p=0.0),
            #nn.PReLU(),
            #nn.Linear(compress_channels * 3, compress_channels * 3),
            #nn.Dropout(p=0.0),
            #nn.PReLU(),
            nn.Linear((n_att_heads * hidden_channels+in_channels) * 3, 2)
        )

    def bn(self, i, x):
        num_nodes, num_channels = x.size()

        x = x.view(-1, num_channels)
        x = getattr(self, f'bn{i}')(x)
        x = x.view(num_nodes, num_channels)
        return x

    def forward(self, x, mask=None):
        x0 = x
        x1 = self.bn(1, self.conv1(x0.x, x0.edge_index, x0.edge_weight))
        #nn.PReLU()
        x1 = torch.cat([x1, x0.x],dim=1)
        nn.PReLU
        x2 = self.bn(2, self.conv2(x1, x0.edge_index, x0.edge_weight))
        nn.PReLU()
        x2 = x2+x1
        nn.PReLU()
        x3 = self.bn(3, self.conv3(x2, x0.edge_index, x0.edge_weight))
        nn.PReLU()
        x3 = x3+x2
        nn.PReLU()
        #x = global_sort_pool(x3, x.batch, k=10)
        x_add = global_add_pool(x3, x.batch)
        x_mean = global_mean_pool(x3, x.batch)
        x_max = global_max_pool(x3, x.batch)
        x = torch.cat([x_add, x_max, x_mean], dim=1)

        mlp = self.layers(x)

        return F.log_softmax(mlp, dim=-1)


device = 'cuda'
#torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n_neurons = 4
model = Net(dataset.dataset[0].num_features, n_neurons, n_neurons).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=30, gamma=0.1,verbose=True)


def train_epoch(logger, loader, model, optimizer):
    model.train()
    for data in loader:
        optimizer.zero_grad()
        data.to(torch.device(device))
        output = model(data)
        weights = torch.zeros(2).cuda()
        (weights[0]) = 1.0
        (weights[1]) = 0.8
        loss = F.nll_loss(output, data.y.view(-1), weight=None)
        pred = model(data).max(dim=1)[1]
        loss.backward()
        optimizer.step()
        logger.update_stats(true=data.y.view(-1).detach().cpu(),
                            pred=pred.detach().cpu(),
                            loss=loss.item(),
                            lr=0, time_used=0, params=0)
        #scheduler.print_lr()

#@torch.no_grad()
def eval_epoch(logger, loader, model):
    model.eval()
    for data in loader:
        data.to(torch.device(device))
        x = data.y.view(-1).type(torch.DoubleTensor)
        pred = model(data).max(dim=1)[1]
        loss = 0
        logger.update_stats(true=data.y.view(-1).detach().cpu(),
                            pred=pred.detach().cpu(),
                            loss=loss,
                            lr=0, time_used=0, params=0)


def train(loggers, loaders, model, optimizer):
    start_epoch = 0
    logging.info('Start from epoch {}'.format(start_epoch))
    num_splits = len(loggers)
    for cur_epoch in range(start_epoch, max_epoch):
        train_epoch(loggers[0], loaders[0], model, optimizer)

        loggers[0].write_epoch(cur_epoch)
        scheduler.step()
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