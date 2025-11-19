import torch
import torch.nn as nn
import torch.nn.functional as F

class GaitClassifier(nn.Module):
    def __init__(self, input_dim=38, num_classes=7):
        super(GaitClassifier, self).__init__()
        self.layer1 = nn.Linear(input_dim, 128)
        self.dropout1 = nn.Dropout(0.3)
        self.layer2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.3)
        self.output = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = self.dropout1(x)
        x = F.relu(self.layer2(x))
        x = self.dropout2(x)
        x = self.output(x)
        return x