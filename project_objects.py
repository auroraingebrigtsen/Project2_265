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

from project_constants import DEVICE as device
from project_constants import SEED

class Timer():
    '''
    Simple class to time the code execution
    '''
    
    def __init__(self):
        self.timer_started = None
        self.end_time = None
        
    def start_timer(self):
        '''
        Starts the timer
        '''
        self.timer_started = time.time()
        
    def stop_timer(self):
        '''
        Ends the timer and prints to terminal.
        '''
        
        self.end_time = time.time() - self.timer_started
        
        return self.end_time
    
class LocalizationLoss(nn.Module):
    """Custom loss function"""
    def __init__(self):
        super().__init__()
        self.L_a = nn.BCEWithLogitsLoss()  # detection loss
        self.L_b = nn.MSELoss()  # localization loss
        self.L_c = nn.CrossEntropyLoss()  # classification loss

    def forward(self, y_pred, y_true):

        det_pred = y_pred[:, 0]
        bbox_pred = y_pred[:, 1:5]
        class_pred = y_pred[:, 5:]

        det_true = y_true[:, 0]
        bbox_true = y_true[:, 1:5]
        class_true = y_true[:, -1].long()

        L_a = self.L_a(det_pred, det_true)

        object_detected = det_true == 1

        L_b = self.L_b(bbox_pred[object_detected], bbox_true[object_detected])
        L_c = self.L_c(class_pred[object_detected], class_true[object_detected])
        
        #print(y_pred[0])

        return L_a + L_b + L_c, (L_a,L_b,L_c)