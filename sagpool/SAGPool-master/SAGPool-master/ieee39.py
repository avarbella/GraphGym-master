import torch
from torch_geometric.data import DataLoader
from torch_geometric import utils
from networks import  Net, Global_Net
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GINConv
import argparse
from sklearn.metrics import balanced_accuracy_score, f1_score
import os
import warnings


warnings.filterwarnings("ignore", category=UserWarning)
parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=777,
                    help='seed')
parser.add_argument('--batch_size', type=int, default=50,
                    help='batch size')
parser.add_argument('--lr', type=float, default=0.0005,
                    help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0001,
                    help='weight decay')
parser.add_argument('--nhid', type=int, default=128,
                    help='hidden size')
parser.add_argument('--pooling_ratio', type=float, default=1,
                    help='pooling ratio')
parser.add_argument('--dropout_ratio', type=float, default=0.0,
                    help='dropout ratio')
parser.add_argument('--dataset', type=str, default='DD',
                    help='DD/PROTEINS/NCI1/NCI109/Mutagenicity')
parser.add_argument('--epochs', type=int, default=100000,
                    help='maximum number of epochs')
parser.add_argument('--patience', type=int, default=50,
                    help='patience for earlystopping')
parser.add_argument('--pooling_layer_type', type=str, default='GCNConv',
                    help='DD/PROTEINS/NCI1/NCI109/Mutagenicity')

args = parser.parse_args()
args.device = 'cpu'
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    args.device = 'cuda:0'

dataset = torch.load('C:/Users/avarbella/Documents/GraphGym-master/'
                     'GraphGym-master/run/datasets/IEEE39/IEEE39/train_IEEE118_swiss_small_geo.pt')
data_test = torch.load('C:/Users/avarbella/Documents/GraphGym-master/'
                       'GraphGym-master/run/datasets/IEEE39/IEEE39/test_IEEE118_swiss_small_geo.pt')

train_dataset = dataset
val_loader = data_test
train_loader = dataset

args.num_classes = 2
args.num_features = dataset.dataset[0].num_features





#train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
#val_loader = DataLoader(data_test,batch_size=args.batch_size,shuffle=False)
#test_loader = DataLoader(test_set,batch_size=1,shuffle=False)
model = Global_Net(args).to(args.device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


def test(model,loader):
    model.eval()
    correct = 0.
    loss = 0.
    pred_list = []
    correct_list = []
    for data in loader:
        data = data.to(args.device)
        out = model(data)
        pred = out.max(dim=1)[1]
        pred_list.append(pred.cpu().detach().numpy())
        correct_list.append(data.y.cpu().detach().numpy())
        correct += pred.eq(data.y).sum().item()
        loss += F.nll_loss(out, data.y, reduction='sum').item()
        #f1 = f1_score(y.cpu().detach().numpy(), pred.cpu().detach().numpy())
        #bal_acc = balanced_accuracy_score(y.cpu().detach().numpy(), pred.cpu().detach().numpy())

    return correct / len(loader.dataset), loss / len(loader.dataset)#,f1,bal_acc


min_loss = 1e10
patience = 0

for epoch in range(args.epochs):
    model.train()
    for i, data in enumerate(train_loader):
        data = data.to(args.device)
        out = model(data)
        t = data.y
        loss = F.nll_loss(out, data.y)
        #print("Training loss:{}".format(loss.item()))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    val_acc, val_loss = test(model, val_loader)
    train_acc, train_loss = test(model, train_loader)
    print("Validation loss:\t{}\taccuracy:\t{}\t".format(val_loss, val_acc))
    print("Training loss:\t{}\taccuracy:\t{}\t".format(train_loss, train_acc))
    if val_loss < min_loss:
        torch.save(model.state_dict(), 'latest.pth')
        print("Model saved at epoch{}".format(epoch))
        min_loss = val_loss
        patience = 0
    else:
        patience += 1
    if patience > args.patience:
        break 

model = Global_Net(args).to(args.device)
model.load_state_dict(torch.load('latest.pth'))
#test_acc,test_loss = test(model,test_loader)
#print("Test accuarcy:{}".fotmat(test_acc))
