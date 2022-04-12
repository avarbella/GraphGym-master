import os
import torch
from torch.utils.data import ConcatDataset





test39 = torch.load('C:/Users/avarbella/Documents/GraphGym-master/'
                       'GraphGym-master/run/datasets/IEEE39/IEEE39/edgeattr_test.pt')
test118 = torch.load('C:/Users/avarbella/Documents/GraphGym-master/'
                       'GraphGym-master/run/datasets/IEEE39/IEEE39/edgeattr118_test.pt')
testswiss = torch.load('C:/Users/avarbella/Documents/GraphGym-master/'
                       'GraphGym-master/run/datasets/IEEE39/IEEE39/edgeattrswiss_test.pt')

test = ConcatDataset((test39, test118, testswiss))
torch.save(test, 'test.pt')

train39 = torch.load('C:/Users/avarbella/Documents/GraphGym-master/'
                     'GraphGym-master/run/datasets/IEEE39/IEEE39/edgeattr_train.pt')
train118 = torch.load('C:/Users/avarbella/Documents/GraphGym-master/'
                     'GraphGym-master/run/datasets/IEEE39/IEEE39/edgeattr118_train.pt')
trainswiss = torch.load('C:/Users/avarbella/Documents/GraphGym-master/'
                     'GraphGym-master/run/datasets/IEEE39/IEEE39/edgeattrswiss_train.pt')

train = ConcatDataset((train39, train118, trainswiss))
torch.save(train, 'train.pt')


