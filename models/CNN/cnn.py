import os 
import sys
import pandas as pd 
import numpy as np 

import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

PROJECT_PATH =  os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
print(PROJECT_PATH)

sys.path.append(PROJECT_PATH)
from features.feature_extractor import FeatureExtractor

class CNN_MODEL(nn.Module):
    def __init__(self):
        super(CNN_MODEL, self).__init__() 
        
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )        
        self.adpater=  nn.AdaptiveAvgPool2d((1, 1))  
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 8),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(8,1),
        )
    def forward(self,x):
        # b,c,coef_dim, time= x.shape
        # print(f"1 {x.shape=} ")
        x = self.conv(x) # batch = 1 Channel =128 ,coef =13 , time variable  
        # print(f"2 {x.shape=}")
        x = self.adpater(x)
        x = x.view(x.size(0), -1)
        # print(f"3 {x.shape=}")
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        # print(f"4 {x.shape=}")
        x = torch.sigmoid(x)
        # print(f"5 {x.shape=}")
        return x


# if __name__ == "__main__":
#     # Create model
#     model = CNN_MODEL()
    
#     # Test with variable-length inputs
#     test_input1 = torch.randn(1, 1, 13, 101)  # Audio 1
#     test_input2 = torch.randn(1, 1, 13, 299)  # Audio 2
    
#     output1 = model(test_input1)
#     output2 = model(test_input2)
    
#     print(f"{output1=}")
#     print(f"{output2=}")
#     print(f"Output 1 shape: {output1.shape}")  # Should be torch.Size([1, 2])
#     print(f"Output 2 shape: {output2.shape}")  # Should be torch.Size([1, 2])
    
#     # Count parameters
#     total_params = sum(p.numel() for p in model.parameters())
#     print(f"Total parameters: {total_params:,}")
