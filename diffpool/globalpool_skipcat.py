import torch
import logging
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GeneralConv,global_max_pool, GATConv,TransformerConv, global_mean_pool, global_sort_pool,global_add_pool
from graphgym.logger import setup_printing, create_logger
from torch.utils.data import DataLoader
import neptune.new as neptune
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, auc

run = neptune.init(
    project="annavarb/Cascades-GNN",
    api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIyMTY2NjYzNi1kZTFjLTQ5NzMtOTQyZi1hMzhlNjg3OTM5ODcifQ==',
    tags='Edgeattr_all, 2hops, simple Gat conv'
)


def is_eval_epoch(cur_epoch, eval_period, max_epoch):
    """Determines if the model should be evaluated at the current epoch."""
    return (
            (cur_epoch + 1) % eval_period == 0 or
            cur_epoch == 0 or
            (cur_epoch + 1) == max_epoch
    )


eval_period = 5
max_epoch = 100
data_test = torch.load('C:/Users/avarbella/Documents/GraphGym-master/'
                     'GraphGym-master/run/datasets/IEEE39/IEEE39/test.pt')

dataset = torch.load('C:/Users/avarbella/Documents/GraphGym-master/'
                     'GraphGym-master/run/datasets/IEEE39/IEEE39/train.pt')


val_loader = data_test
train_loader = dataset
datasets = [train_loader, val_loader]


class Net(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        n_att_heads = 3

        self.conv1 = GeneralConv(in_channels, hidden_channels, in_edge_channels=4, heads=n_att_heads)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels * n_att_heads)
        compress_channels = 4
        self.conv2 = GeneralConv(n_att_heads * hidden_channels + in_channels,
                             n_att_heads * hidden_channels + in_channels, in_edge_channels=4)
        self.bn2 = torch.nn.BatchNorm1d(n_att_heads * hidden_channels + in_channels)
        self.conv3 = GATConv(n_att_heads * hidden_channels + 2*in_channels + hidden_channels,
                             n_att_heads * hidden_channels + 2* in_channels + hidden_channels, edge_dim=4)
        self.bn3 = torch.nn.BatchNorm1d(n_att_heads * hidden_channels + 2* in_channels + hidden_channels)
        self.layers = nn.Sequential(
            #nn.Flatten(),
            nn.Linear((n_att_heads * hidden_channels + in_channels) * 3 * 2,
                      (n_att_heads * hidden_channels+ in_channels) * 3),
            nn.Dropout(p=0.0),
            nn.PReLU(),
            nn.Linear((n_att_heads * hidden_channels + in_channels) * 3, 2)
        )

    def bn(self, i, x):
        num_nodes, num_channels = x.size()

        x = x.view(-1, num_channels)
        x = getattr(self, f'bn{i}')(x)
        x = x.view(num_nodes, num_channels)
        return x

    def forward(self, x, mask=None):
        x0 = x
        x1 = self.bn(1, self.conv1(x0.x, x0.edge_index, x0.edge_attr))
        #nn.PReLU()
        x1 = torch.cat([x1, x0.x], dim=1)
        nn.PReLU
        x2 = self.bn(2, self.conv2(x1, x0.edge_index, x0.edge_attr))
        #nn.PReLU()
        x2 = torch.cat([x2, x1], dim=1)
        nn.PReLU()
        #x3 = self.bn(3, self.conv3(x2, x0.edge_index, x0.edge_attr))
        ##nn.PReLU()
        #x3 = torch.cat([x3, x2], dim=1)
        #nn.PReLU()

        x_add = global_add_pool  (x2, x.batch)
        x_mean = global_mean_pool(x2, x.batch)
        x_max = global_max_pool  (x2, x.batch)
        x = torch.cat([x_add, x_max, x_mean], dim=1)
        nn.PReLU()

        mlp = self.layers(x)

        return F.log_softmax(mlp, dim=-1)


device = 'cuda'
#torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n_neurons = 16
model = Net(3, n_neurons, n_neurons).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=9e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)


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
        run["train/batch/loss"].log(loss.item())
        run["train/batch/f1"].log(f1_score(data.y.view(-1).detach().cpu(), pred.detach().cpu(), zero_division=1))
        run["train/batch/acc"].log(accuracy_score(data.y.view(-1).detach().cpu(), pred.detach().cpu()))
        run["train/batch/prec"].log(precision_score(data.y.view(-1).detach().cpu(), pred.detach().cpu(), zero_division=1))
        run["train/batch/rec"].log(recall_score(data.y.view(-1).detach().cpu(), pred.detach().cpu()))
        loss.backward()
        optimizer.step()
        logger.update_stats(true=data.y.view(-1).detach().cpu(),
                            pred=pred.detach().cpu(),
                            loss=loss.item(),
                            lr=0, time_used=0, params=0)

#@torch.no_grad()
def eval_epoch(logger, loader, model):
    model.eval()
    for data in loader:
        data.to(torch.device(device))
        x = data.y.view(-1).type(torch.DoubleTensor)
        output = model(data)
        pred = output.max(dim=1)[1]
        #loss = F.nll_loss(output, data.y.view(-1), weight=None)
        logger.update_stats(true=data.y.view(-1).detach().cpu(),
                            pred=pred.detach().cpu(),
                            loss=0,
                            lr=0, time_used=0, params=0)
        #run["val/batch/loss"].log(loss.item())
        run["val/batch/f1"].log(f1_score(data.y.view(-1).detach().cpu(), pred.detach().cpu(), zero_division=1))
        run["val/batch/acc"].log(accuracy_score(data.y.view(-1).detach().cpu(), pred.detach().cpu()))
        run["val/batch/prec"].log(precision_score(data.y.view(-1).detach().cpu(), pred.detach().cpu(), zero_division=1))
        run["val/batch/rec"].log(recall_score(data.y.view(-1).detach().cpu(), pred.detach().cpu()))


def train(loggers, loaders, model, optimizer):
    start_epoch = 0
    logging.info('Start from epoch {}'.format(start_epoch))
    num_splits = len(loggers)
    for cur_epoch in range(start_epoch, max_epoch):
        train_epoch(loggers[0], loaders[0], model, optimizer)
        scheduler.step()
        print(scheduler.get_last_lr())
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
