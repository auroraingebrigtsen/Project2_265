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

from project2_constants import DEVICE as device
from project2_constants import SEED

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
    """Custom loss function for a object localization task"""
    def __init__(self):
        super().__init__()
        self.L_a = nn.BCEWithLogitsLoss()  # detection loss
        self.L_b = nn.MSELoss()  # localization loss
        self.L_c = nn.CrossEntropyLoss()  # classification loss
        self.L_c_binary = nn.BCEWithLogitsLoss() # binary classification loss

    def forward(self, y_pred, y_true):
        det_pred = y_pred[:, 0]
        bbox_pred = y_pred[:, 1:5]
        class_pred = y_pred[:, 5:]

        binary_classification = class_pred.shape[1] == 1

        det_true = y_true[:, 0]
        bbox_true = y_true[:, 1:5]
        class_true = y_true[:, -1] if binary_classification else y_true[:, -1].long()

        L_a = self.L_a(det_pred, det_true)

        object_detected = det_true == 1

        L_b = self.L_b(bbox_pred[object_detected], bbox_true[object_detected])

        predicted_objects = class_pred[object_detected].squeeze(1) if binary_classification else class_pred[object_detected]
        L_c = self.L_c_binary if binary_classification else self.L_c
        L_c = L_c(predicted_objects, class_true[object_detected])
        
        return L_a + L_b + L_c, (L_a,L_b,L_c)
    


class DetectionLoss(nn.Module):
    """Custom loss function for a object detection task"""
    def __init__(self):
        super().__init__()
        self.Localization_loss = LocalizationLoss()

    def forward(self, y_pred, y_true):
        y_pred_reshaped = y_pred.permute(0,2,3,1)
        y_pred_reshaped = y_pred_reshaped.reshape(-1, y_pred_reshaped.size(-1))

        y_true_reshaped = y_true.permute(0,2,3,1)
        y_true_reshaped = y_true_reshaped.reshape(-1, y_true_reshaped.size(-1))

        loss, loss_tuples = self.Localization_loss(y_pred_reshaped, y_true_reshaped)

        return loss, loss_tuples