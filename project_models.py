import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import torch.optim as optim
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import random_split
from collections import Counter
import seaborn as sns
from sklearn.metrics import confusion_matrix
from torchvision.utils import draw_bounding_boxes
from torchvision.ops import box_convert, box_iou
from typing import Sequence
from math import floor, sqrt
import time
import random

from project_functions import fc_size
from project_constants import DEVICE as device
from project_constants import SEED


class LocCNN1(nn.Module):
    def __init__(self, input_size):
        self.input_size = input_size
        
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1, device=device, dtype=torch.double)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1, device=device, dtype=torch.double)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, device=device, dtype=torch.double)
        self.fc1 = nn.Linear(fc_size(self.input_size, self.children()), 15, device=device, dtype=torch.double)
        self.flatten = nn.Flatten()

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.pool1(out)
        out = F.relu(self.conv2(out))
        out = self.pool2(out)
        out = F.relu(self.conv3(out))
        out = self.flatten(out)
        out = self.fc1(out)
        return out
    
class LocCNN2(nn.Module):
    def __init__(self, input_size):
        self.input_size = input_size
        
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1, device=device, dtype=torch.double)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1, device=device, dtype=torch.double)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,2), stride=1, padding=1, device=device, dtype=torch.double)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, device=device, dtype=torch.double)
        self.pool4 = nn.MaxPool2d(kernel_size=2)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1, device=device, dtype=torch.double)
        self.conv6 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1, device=device, dtype=torch.double)
        self.fc1 = nn.Linear(fc_size(self.input_size, self.children()), 100, device=device, dtype=torch.double)
        self.fc2 = nn.Linear(100, 15, device=device, dtype=torch.double)
        self.flatten = nn.Flatten()

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.pool1(out)
        out = F.relu(self.conv2(out))
        out = self.pool2(out)
        out = F.relu(self.conv3(out))
        out = self.pool3(out)
        out = F.relu(self.conv4(out))
        out = self.pool4(out)
        out = F.relu(self.conv5(out))
        out = F.relu(self.conv6(out))
        out = self.flatten(out)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out
    
class LocCNN3(nn.Module):
    def __init__(self, input_size):
        self.input_size = input_size
        
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1, device=device, dtype=torch.double)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1, device=device, dtype=torch.double)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,2), stride=1, padding=1, device=device, dtype=torch.double)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, device=device, dtype=torch.double)
        self.pool4 = nn.MaxPool2d(kernel_size=2)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1, device=device, dtype=torch.double)
        self.conv6 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1, device=device, dtype=torch.double)
        self.fc1 = nn.Linear(fc_size(self.input_size, self.children()), 100, device=device, dtype=torch.double)
        self.fc2 = nn.Linear(100, 15, device=device, dtype=torch.double)
        self.flatten = nn.Flatten()

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.dropout(out, p=0.1, training=self.training)
        out = self.pool1(out)
        out = F.relu(self.conv2(out))
        out = F.dropout(out, p=0.1, training=self.training)
        out = self.pool2(out)
        out = F.relu(self.conv3(out))
        out = F.dropout(out, p=0.1, training=self.training)
        out = self.pool3(out)
        out = F.relu(self.conv4(out))
        out = F.dropout(out, p=0.1, training=self.training)
        out = self.pool4(out)
        out = F.relu(self.conv5(out))
        out = F.dropout(out, p=0.1, training=self.training)
        out = F.relu(self.conv6(out))
        out = F.dropout(out, p=0.1, training=self.training)
        out = self.flatten(out)
        out = F.relu(self.fc1(out))
        out = F.dropout(out, p=0.5, training=self.training)
        out = self.fc2(out)
        return out
    
class LocCNN4(nn.Module):
    def __init__(self, input_size):
        self.input_size = input_size
        
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1, device=device, dtype=torch.double)
        self.pool1 = nn.MaxPool2d(kernel_size=2, return_indices=True)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1, device=device, dtype=torch.double)
        self.pool2 = nn.MaxPool2d(kernel_size=2, return_indices=True)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, device=device, dtype=torch.double)
        self.flatten = nn.Flatten()
        
        self.fc1 = nn.Linear(fc_size(self.input_size, self.children()), fc_size(self.input_size, self.children()), device=device, dtype=torch.double)
        self.fc2 = nn.Linear(fc_size(self.input_size, self.children()), fc_size(self.input_size, self.children()), device=device, dtype=torch.double)
        self.unflatten = nn.Unflatten(1,(64,12,15))
        
        self.deconv1 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1, device=device, dtype=torch.double)
        self.unpool1 = nn.MaxUnpool2d(kernel_size=2)
        self.deconv2 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1, device=device, dtype=torch.double)
        self.unpool2 = nn.MaxUnpool2d(kernel_size=2)
        self.deconv3 = nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=3, stride=1, padding=1, device=device, dtype=torch.double)
        self.fc3 = nn.Linear(60*48, 15, device=device, dtype=torch.double)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out, indices1 = self.pool1(out)
        out = F.relu(self.conv2(out))
        out, indices2 = self.pool2(out)
        out = F.relu(self.conv3(out))

        out = self.flatten(out)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.unflatten(out)
        out = F.relu(self.deconv1(out))
        out = self.unpool1(out, indices2)
        out = F.relu(self.deconv2(out))
        out = self.unpool2(out, indices1)
        out = F.relu(self.deconv3(out))
        out = self.flatten(out)
        out = self.fc3(out)
        
        return out
    
class LocCNN5(nn.Module):
    def __init__(self, input_size):
        self.input_size = input_size
        
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=7, stride=1, padding=1, device=device, dtype=torch.double)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=7, stride=1, padding=1, device=device, dtype=torch.double)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(8,7), stride=1, padding=1, device=device, dtype=torch.double)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, device=device, dtype=torch.double)
        self.flatten = nn.Flatten()
        
        self.fc1 = nn.Linear(fc_size(self.input_size, self.children()), 1000, device=device, dtype=torch.double)
        self.fc2 = nn.Linear(1000, 1000, device=device, dtype=torch.double)
        self.fc3 = nn.Linear(1000, 15, device=device, dtype=torch.double)

    def forward(self, x):
        #print(x.shape)
        out = F.relu(self.conv1(x))
        out = F.dropout(out, p=0.1, training=self.training)
        #print(out.shape)

        out = self.pool1(out)
        #print(out.shape)

        out = F.relu(self.conv2(out))
        out = F.dropout(out, p=0.1, training=self.training)
        #print(out.shape)
        out = self.pool2(out)
        #print(out.shape)
        out = F.relu(self.conv3(out))
        out = F.dropout(out, p=0.1, training=self.training)
        #print(out.shape)
        out = self.pool3(out)
        #print(out.shape)

        out = F.relu(self.conv4(out))
        out = F.dropout(out, p=0.1, training=self.training)
        #print(out.shape)
 
        out = self.flatten(out)
        #print(out.shape)
        
        out = F.relu(self.fc1(out))
        out = F.dropout(out, p=0.5, training=self.training)
        #print(out.shape)
        out = F.relu(self.fc2(out))
        out = F.dropout(out, p=0.5, training=self.training)
        #print(out.shape)
        out = self.fc3(out)
        #print(out.shape)
        return out
    
class DetCNN1(nn.Module):
    def __init__(self):
        
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1, device=device, dtype=torch.double)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1, device=device, dtype=torch.double)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,2), stride=1, padding=1, device=device, dtype=torch.double)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=1, device=device, dtype=torch.double)
        self.pool4 = nn.MaxPool2d(kernel_size=2)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1, device=device, dtype=torch.double)
        self.conv6 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1, device=device, dtype=torch.double)
        self.conv7 = nn.Conv2d(in_channels=32, out_channels=6, kernel_size=3, stride=1, padding=1, device=device, dtype=torch.double)


    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.pool1(out)
        out = F.relu(self.conv2(out))
        out = self.pool2(out)
        out = F.relu(self.conv3(out))
        out = self.pool3(out)
        out = F.relu(self.conv4(out))
        out = self.pool4(out)
        out = F.relu(self.conv5(out))
        out = F.relu(self.conv6(out))
        out = self.conv7(out)

        return out
    

class DetCNN2(nn.Module):
    def __init__(self):

        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2, device=device, dtype=torch.double)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2, device=device, dtype=torch.double)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,4), stride=1, padding=1, device=device, dtype=torch.double)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(4,3), stride=1, padding=1, device=device, dtype=torch.double)
        self.pool4 = nn.MaxPool2d(kernel_size=2)
        self.conv5 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1, device=device, dtype=torch.double)
        self.conv6 = nn.Conv2d(in_channels=16, out_channels=6, kernel_size=3, stride=1, padding=1, device=device, dtype=torch.double)


    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.pool1(out)
        out = F.relu(self.conv2(out))
        out = self.pool2(out)
        out = F.relu(self.conv3(out))
        out = self.pool3(out)
        out = F.relu(self.conv4(out))
        out = self.pool4(out)
        out = F.relu(self.conv5(out))
        out = self.conv6(out)

        return out