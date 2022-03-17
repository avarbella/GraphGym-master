import torch
from graphgym.loader import create_dataset, create_loader
from deepsnap.dataset import GraphDataset

model = torch.load('C:/GNN/modelGAT_model.pth')

model.load_state_dict(torch.load('C:/GNN/modelGAT.pth'))
dataset_raw = torch.load('C:/Users/avarbella/Documents/GraphGym-master/'              
                                'GraphGym-master/run/datasets/IEEE39/IEEE39/test_dataset.pt')
#print(next(iter(dataset_raw)))
datasets = GraphDataset.pyg_to_graphs(dataset_raw)
loaders = create_loader(datasets)

#for batch in loaders:
with torch.no_grad():
    for data in dataset_raw:
        #model.eval()

        pred = model(data)
    #loss, pred_score = compute_loss(pred, true)
