import torch
import torch.onnx as onnx
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_max_pool, GATConv,TransformerConv, global_mean_pool,global_add_pool
import neptune.new as neptune
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, auc
import netron

device = 'cuda'
class Net(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(Net, self).__init__()
        n_att_heads = 4
        self.conv1 = TransformerConv(in_channels, hidden_channels, edge_dim=4, heads=n_att_heads)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels * n_att_heads)
        self.conv2 = TransformerConv(n_att_heads * hidden_channels,n_att_heads * hidden_channels, edge_dim=4)
        self.bn2 = torch.nn.BatchNorm1d(n_att_heads * hidden_channels)
        self.conv3 = TransformerConv(n_att_heads * hidden_channels, n_att_heads * hidden_channels, edge_dim=4)
        self.bn3 = torch.nn.BatchNorm1d(n_att_heads * hidden_channels)
        self.layers = nn.Sequential(
            #nn.Flatten(),
            nn.Linear(n_att_heads * hidden_channels*3 , n_att_heads * hidden_channels*3),
            nn.Dropout(p=0.0),
            nn.PReLU(),

            nn.Linear(n_att_heads * hidden_channels*3 , 2)
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
        nn.PReLU()
        x2 = self.bn(2, self.conv2(x1, x0.edge_index, x0.edge_attr))
        nn.PReLU()
        x3 = self.bn(3, self.conv3(x2, x0.edge_index, x0.edge_attr))
        nn.PReLU()

        x_add = global_add_pool(x3, x.batch)
        x_mean = global_mean_pool(x3, x.batch)
        x_max = global_max_pool(x3, x.batch)
        x = torch.cat([x_add, x_max, x_mean], dim=1)
        #x,_ = x3.max(dim=1)

        mlp = self.layers(x)
        return F.log_softmax(mlp, dim=-1)


model = Net(3,16).cuda()

model.load_state_dict(torch.load('model.pt'))

#run = neptune.init(
#    project="annavarb/Cascades-GNN",
#    api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIyMTY2NjYzNi1kZTFjLTQ5NzMtOTQyZi1hMzhlNjg3OTM5ODcifQ==',
#    tags='transfer learn on IEEE24'
#)
data = torch.load('C:/Users/avarbella/Documents/GraphGym-master/'
                             'GraphGym-master/IEEE39/IEEE24_train.pt')
data_test = torch.load('C:/Users/avarbella/Documents/GraphGym-master/'
                             'GraphGym-master/IEEE39/IEEE24.pt')
onnx.export(model,data,'gnn.onnx')
netron.start('gnn.onnx')
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=5e-4)
def train_epoch( loader, model, optimizer):
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
       #run["train/batch/loss"].log(loss.item())
       #run["train/batch/f1"].log(f1_score(data.y.view(-1).detach().cpu(), pred.detach().cpu(), zero_division=1))
       #run["train/batch/acc"].log(accuracy_score(data.y.view(-1).detach().cpu(), pred.detach().cpu()))
       #run["train/batch/prec"].log(precision_score(data.y.view(-1).detach().cpu(), pred.detach().cpu(), zero_division=1))
       #run["train/batch/rec"].log(recall_score(data.y.view(-1).detach().cpu(), pred.detach().cpu()))
        loss.backward()
        optimizer.step()

def test_epoch( loader, model):
    model.eval()
    for data in loader:
        data.to(torch.device(device))
        x = data.y.view(-1).type(torch.DoubleTensor)
        pred = model(data).max(dim=1)[1]
        loss = 0
       # run["test/batch/f1"].log(f1_score(data.y.view(-1).detach().cpu(), pred.detach().cpu(), zero_division=1))
       # run["test/batch/acc"].log(accuracy_score(data.y.view(-1).detach().cpu(), pred.detach().cpu()))
       # run["test/batch/prec"].log(precision_score(data.y.view(-1).detach().cpu(), pred.detach().cpu(), zero_division=1))
       # run["test/batch/rec"].log(recall_score(data.y.view(-1).detach().cpu(), pred.detach().cpu()))

def is_eval_epoch(cur_epoch, eval_period,max_epoch):
    """Determines if the model should be evaluated at the current epoch."""
    return (
            (cur_epoch + 1) % eval_period == 0 or
            cur_epoch == 0 or
            (cur_epoch + 1) == max_epoch
    )

eval_period = 10
max_epoch = 50
def train( loader, model, optimizer):
    start_epoch = 0

    for cur_epoch in range(start_epoch, max_epoch):
        train_epoch(loader, model, optimizer)


        if is_eval_epoch(cur_epoch, eval_period, max_epoch):
            #for i in range(1, num_splits):
            test_epoch(data_test, model)

train(data, model, optimizer)
