import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class embedding_classifier(nn.Module):
    def __init__(self):
        super(embedding_classifier, self).__init__()
        
        ## softmax classifier
        fc1 = nn.Linear(in_features)
