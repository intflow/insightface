import numpy as np
import torch
use_cuda = torch.cuda.is_available()
cpu_device = torch.device('cpu')
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class embedding_classifier(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(embedding_classifier, self).__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes
        
        ## softmax classifier
        fc1 = nn.Linear(in_features=self.input_shape, out_features=1024, bias=True)
        BN1 = nn.BatchNorm1d(num_features=1024)
        relu1 = nn.ReLU()
        dropout1 = nn.Dropout(p=0.5)

        fc2 = nn.Linear(in_features=1024, out_features=1024, bias=True)
        BN2 = nn.BatchNorm1d(num_features=1024)
        relu2 = nn.ReLU()
        dropout2 = nn.Dropout(p=0.5)

        out = nn.Linear(in_features=1024, out_features=self.num_classes)

        self.fc_module = nn.Sequential(
            #layer1
            fc1,
            BN1,
            relu1,
            dropout1,
            
            #layer2
            fc2,
            BN2,
            relu2,
            dropout2,
            #output
            out
        )

        # if use_cuda:
        #     self.fc_module = self.fc_module.cuda()
        self.fc_module = self.fc_module.to(cpu_device)

    def forward(self, input_data):
        # input_data = torch.tensor(input_data).to(cpu_device)
        # input_data.clone().detach()
        # if use_cuda:
        #     input_data = input_data.cuda()
        input_data = torch.tensor(input_data, device=cpu_device)
        out = self.fc_module(input_data)
        
        return out.cuda()



