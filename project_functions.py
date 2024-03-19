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
from torch.utils.data import TensorDataset

from project_objects import Timer
from project_constants import DEVICE as device
from project_constants import SEED

def count_instances(data, data_name=None) -> None:
    """Counts the number of instances of each class in a dataset"""
    counter = Counter([99 if label[0] == 0 else int(label[-1]) for _, label in data])
    sorted_counter = dict(sorted(counter.items()))
    if data_name is not None:
        print(f'Class distribution in {data_name}')
    for key, value in sorted_counter.items():
        print(f'{key}: {value}') 
        
def plot_images(data):
    _, axes = plt.subplots(nrows=2, ncols=6, figsize=(8,3))

    for i, ax in enumerate(axes.flat): 

        if i == 10:
            img = next(img for img, label in data if int(label[0]) == 0)
            img = img.numpy().transpose((1, 2, 0))
            ax.imshow(img, cmap='gray')
            ax.set_title('None')
            ax.axis('off')
            continue

        if i == 11:
            ax.axis('off')
            continue
        
        img, bbox = next((img, label[1:5]) for img, label in data if int(label[-1]) == i)
        img_height, img_width = img.shape[-2], img.shape[-1]

        img = img.clone()
        img = (img * 255).byte()

        bbox = bbox.clone()
        bbox[0] *= img_width
        bbox[1] *= img_height
        bbox[2] *= img_width
        bbox[3] *= img_height

        bbox = bbox.type(torch.uint8)

        converted_bbox = box_convert(bbox, in_fmt='cxcywh', out_fmt='xyxy')

        img_with_bbox = draw_bounding_boxes(img, converted_bbox.unsqueeze(0), colors='lightgreen')
        img_with_bbox  = img_with_bbox.numpy().transpose((1, 2, 0))
        ax.imshow(img_with_bbox, cmap='gray')
        ax.set_title(i)
        ax.axis('off')

def plot_class(data:torch.tensor, class_label:int, start_idx:int=0) -> None:
    """Plots a subplot with 10 images from a given class, starting at a chosen index"""
    class_images = [img for img, label in data if int(label[-1]) == class_label]
    bboxes = [label[1:5] for img, label in data if int(label[-1]) == class_label]
    _, axes = plt.subplots(nrows=2, ncols=5, figsize=(8,3))

    for i, ax in enumerate(axes.flat):
        


        idx = start_idx + i
        img = class_images[idx].clone()
        bbox = bboxes[idx].clone()

        img_height, img_width = data[0][0].shape[-2], data[0][0].shape[-1]

        img = (img * 255).byte()

        bbox[0] *= img_width
        bbox[1] *= img_height
        bbox[2] *= img_width
        bbox[3] *= img_height

        bbox = bbox.type(torch.uint8)

        converted_bbox = box_convert(bbox, in_fmt='cxcywh', out_fmt='xyxy')

        img_with_bbox = draw_bounding_boxes(img, converted_bbox.unsqueeze(0), colors='lightgreen')
        img_with_bbox  = img_with_bbox.numpy().transpose((1, 2, 0))
        ax.imshow(img_with_bbox, cmap='gray')
        plt.suptitle(f'CLASS {class_label} - Image {start_idx} to {idx}')
        ax.axis('off')

    plt.show()

def fc_size(size, layers):
    input_size = size

    for layer in layers:
        if isinstance(layer, torch.nn.modules.linear.Linear):
            break
        
        elif isinstance(layer, torch.nn.modules.conv.Conv2d) or isinstance(layer, torch.nn.modules.pooling.MaxPool2d):
            input_size = get_output_size(input_size ,layer)     
        
    network_size = input_size[0] * input_size[1] * input_size[2]
    return int(network_size)
        

def get_output_size(input_size:tuple, layer:nn.Module):
    padding = int_to_pair(layer.padding)
    stride = int_to_pair(layer.stride)
    kernel = int_to_pair(layer.kernel_size)

    channels = input_size[2]
    
    width = input_size[0]
    height = input_size[1]

    width_out = (width+2*padding[0]-kernel[0])/stride[0] + 1
    height_out = (height+2*padding[1]-kernel[1])/stride[1] + 1
    
    if isinstance(layer, torch.nn.modules.conv.Conv2d):
        channels = layer.out_channels

    return (width_out, height_out, channels)

def int_to_pair(n):
    """
    Return `(n, n)` if `n` is an int or `n` if it is already a tuple of length 2
    """
    if not isinstance(n, Sequence):
        return (int(n), int(n))
    elif len(n) == 1:
        return (int(n[0]), int(n[0]))
    elif len(n) == 2:
        return ( int(n[0]), int(n[1]) )
    else:
        raise ValueError("Please give an int or a pair of int")
    
def compute_performance(model, loader):
    '''
    Function that uses a model to predict and calculate accuracy
    '''
    model.eval()
    correct = 0
    total = 0
    iou_sum = 0

    with torch.inference_mode():
        for imgs, labels in loader:
            imgs = imgs.to(device=device, dtype=torch.double)
            labels = labels.to(device=device, dtype=torch.double)

            outputs = model(imgs)

            det_pred = F.sigmoid(outputs[:, 0])
            object_detected = det_pred > 0.5

            _, class_pred = torch.max(outputs[:, 5:], dim=1)

            det_true = labels[:, 0].int()
            class_true = labels[:, -1].int()

            total += labels.shape[0]
            correct += ((object_detected == 0) & (det_true == 0)).sum()
            correct += ((object_detected == 1) & (det_true == 1) & (class_pred == class_true)).sum()

            bbox_pred = outputs[:, 1:5]
            bbox_true = labels[:, 1:5]

            converted_bbox_pred = box_convert(bbox_pred[object_detected], in_fmt='cxcywh', out_fmt='xyxy')
            converted_bbox_true = box_convert(bbox_true[object_detected], in_fmt='cxcywh', out_fmt='xyxy')
            
            bbox_iou = box_iou(converted_bbox_pred,converted_bbox_true)
            mean_iou = bbox_iou.diag().sum()
            
            iou_sum += mean_iou.item()
            
    acc =  correct / total
    iou = iou_sum / total

    performance = (acc + iou) / 2
    
    return acc, iou, performance

def plot_loss(train_loss:list, val_loss:list, title:str, save_dir='test_results/', save_model=False) -> None:
    """Plots the training and validation loss"""
    _, ax = plt.subplots()
    ax.plot(np.arange(1,len(train_loss)+1), train_loss, label='Training loss')
    ax.plot(np.arange(1,len(val_loss)+1), val_loss, label='Validation loss')
    ax.set_title(title)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    
    if save_model:
        plt.savefig(save_dir+title+'_train_val_loss', bbox_inches='tight')
        
    plt.show()

def plot_lists(data, loss_names:list, title:str, save_dir='test_results/', save_model=False):
    transposed_data = list(map(list, zip(*data)))
    for i, column in enumerate(transposed_data):
        plt.plot(column, label=f'{loss_names[i]}')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    if save_model:
        plt.savefig(save_dir+title+'_losses_separated', bbox_inches='tight')
        
    plt.show()
    
def train(n_epochs, optimizer, model, loss_fn, train_loader, val_loader):
    
    n_batch_train = len(train_loader)
    n_batch_val = len(val_loader)

    losses_train = []
    losses_val = []
    losses_separated = []
    
    time_spent = 0

    optimizer.zero_grad(set_to_none=True)
    
    #try:
        
    for epoch in range(1, n_epochs + 1):
        
        loss_train = 0
        loss_val = 0
        inner_losses_separated = [0,0,0]

        model.train()
        
        # timer
        timer = Timer()
        timer.start_timer()
        
        for imgs, labels in train_loader:

            imgs = imgs.to(device=device, dtype=torch.double)
            labels = labels.to(device=device, dtype=torch.double)

            outputs = model(imgs)
            
            loss, loss_tuples = loss_fn(outputs, labels)
            loss.backward()
            
            optimizer.step()
            optimizer.zero_grad()

            loss_train += loss.item()
            for i in range(3):
                inner_losses_separated[i] += loss_tuples[i].item()
            
        model.eval()

        with torch.inference_mode(): # <-- Equivalent to no_grad, if no error is provided this is preferred.
            for imgs, labels in val_loader:

                imgs = imgs.to(device=device, dtype=torch.double)
                labels = labels.to(device=device, dtype=torch.double)

                outputs = model(imgs)

                loss, _ = loss_fn(outputs, labels)
                loss_val += loss.item()
            
        losses_train.append(loss_train / n_batch_train)
        losses_val.append(loss_val / n_batch_val)
        losses_separated.append([each/n_batch_train for each in inner_losses_separated])
        
        # Calculates time left of training
        time_spent += timer.stop_timer()
        time_left = (time_spent/epoch)*(n_epochs-epoch)
                
        #if epoch == 1 or epoch % 10 == 0:
        print('{}  |  Epoch {}  |  Training loss {:.3f}'.format(datetime.now().strftime('%H:%M:%S'), epoch, loss_train / n_batch_train))
        print('{}  |  Epoch {}  |  Validation loss {:.3f}'.format(datetime.now().strftime('%H:%M:%S'), epoch, loss_val / n_batch_val))
        #print('{}  |  Epoch {}  |  loss {}'.format(datetime.now().strftime('%H:%M:%S'), epoch, losses_separated[epoch - 1]))

        print(f"Estimated time left: {floor(time_left/60)}m {round(time_left%60)}s")
        
#except Exception as e:
    
    #print(e)
    
#finally:
    
    train_acc, train_iou, train_performance = compute_performance(model, train_loader)
    val_acc, val_iou, val_performance = compute_performance(model, val_loader)
    print(f'Training performance: Accuracy = {train_acc}, IOU = {train_iou}, Overall = {train_performance}')
    print(f'Validation performance: Accuracy = {val_acc}, IOU = {val_iou}, Overall = {val_performance}')

    return losses_train, losses_val, train_performance, val_performance, losses_separated
    
def model_selector(models:list, performances:list):
    """Given a list of models, returns the model that has best accuracy score on validation data"""
    best_model = None
    best_performance = 0

    for idx, model in enumerate(models):
        if performances[idx] > best_performance:
            best_model = model
            best_performance = performances[idx]

    return best_model, best_performance

def predict(model, loader):
    '''
    Function that creates a y and y_pred tensor given a model and a loader
    '''
    model.eval()
    
    y_true = torch.empty(0, device=device)
    y_pred = torch.empty(0, device=device)

    with torch.inference_mode(): # <-- Equivalent to no_grad, if no error is provided this is preferred.
        for imgs, labels in loader:
            
            imgs = imgs.to(device=device, dtype=torch.double) 
            labels = labels.to(device=device)
            outputs = model(imgs)
            
            _, class_pred = torch.max(outputs[:, 5:], dim=1)

            predicted = torch.cat((outputs[:, :5], class_pred.unsqueeze(1)), dim=1)
            
            y_true = torch.cat((y_true, labels), dim=0)
            y_pred = torch.cat((y_pred, predicted.data), dim=0)
                
    return y_true, y_pred

def plot_predictions(dataset, y_true:torch.tensor, y_pred:torch.tensor, label:int=0, start_idx:int=0, save_dir='test_results/', fig_name='new', save_model=False) -> None:
    """Plots things"""
    imgs = [img for img,_ in dataset]
    class_mask = y_true[:, -1] == label
    class_imgs = [img for idx, img in enumerate(imgs) if class_mask[idx]]
    class_true, class_pred = y_true[class_mask], y_pred[class_mask]
    
    true_bboxes = [label[1:5] for label in class_true]
    pred_bboxes = [label[1:5] for label in class_pred]

    _, axes = plt.subplots(nrows=2, ncols=5, figsize=(8,3))

    for i, ax in enumerate(axes.flat):

        idx = start_idx + i
        img = class_imgs[idx].clone()

        img_height, img_width = img.shape[-2], img.shape[-1]
        img = (img * 255).byte()

        if int(class_true[idx][0]) == 1:
            true_bbox = true_bboxes[idx].clone() # TODO repetiv kode, lage en funksjon
            true_bbox[0] *= img_width
            true_bbox[1] *= img_height
            true_bbox[2] *= img_width
            true_bbox[3] *= img_height
            
            true_bbox = true_bbox.type(torch.uint8)
            true_bbox_converted = box_convert(true_bbox, in_fmt='cxcywh', out_fmt='xyxy')
            true_bbox_converted = true_bbox_converted.unsqueeze(0)

            img = draw_bounding_boxes(img, true_bbox_converted, colors='lightgreen')

        if F.sigmoid(class_pred[idx][0]) > 0.5:
            pred_bbox = pred_bboxes[idx].clone()
            pred_bbox[0] *= img_width
            pred_bbox[1] *= img_height
            pred_bbox[2] *= img_width
            pred_bbox[3] *= img_height
            
            pred_bbox = pred_bbox.type(torch.uint8)
            pred_bbox_converted = box_convert(pred_bbox, in_fmt='cxcywh', out_fmt='xyxy')
            pred_bbox_converted = pred_bbox_converted.unsqueeze(0)

            img = draw_bounding_boxes(img, pred_bbox_converted, colors='red')
            
        img = img.numpy().transpose((1, 2, 0))
        ax.imshow(img, cmap='gray')
        ax.set_title(f'Pred: {int(class_pred[idx][-1])}')
        plt.suptitle(f'True label: {label} - Image {start_idx} to {idx}')
        ax.axis('off')
        
    if save_model:
        plt.savefig(save_dir+fig_name+'_bbox_pred', bbox_inches='tight')
        
    plt.show()


def global_to_local(labels_list:list, grid_dimension:tuple):
    '''
    Transfers one list of tensors to local values
    '''
   
    x = grid_dimension[1]
    y = grid_dimension[0]
    vector_length = grid_dimension[2]

    local_tensor = torch.zeros(y,x,vector_length)
    
    x_grid_cells = [(i+1)/x for i in range(x)]
    y_grid_cells = [(i+1)/y for i in range(y)]

    for label in labels_list:
        label = label.clone()
        x_cell = next(i for i, cell in enumerate(x_grid_cells) if label[1] < cell)
        y_cell = next(i for i, cell in enumerate(y_grid_cells) if label[2] < cell)

        if x_cell != 0:
            label[1] -= x_grid_cells[x_cell - 1]
        if y_cell != 0:
            label[2] -= y_grid_cells[y_cell - 1]

        label[1] *= x
        label[3] *= x
        label[2] *= y
        label[4] *= y

        local_tensor[y_cell][x_cell] = label

    return local_tensor

def prepare_labels(label_dataset:list, grid_dimension:tuple):
    '''
    Iterates through each listed tensor, transforms from global to local coordinates, and stacks them into a new tensor.
    '''

    new_tensor = torch.stack([global_to_local(label, grid_dimension) for label in label_dataset])
    new_tensor = new_tensor.permute(0, 3, 1, 2) 

    return new_tensor

def merge_datasets(d1, d2):
    '''
    Combines the new labels with the image data.
    '''

    return TensorDataset(d1[:][0],d2[:])

def local_to_global(labels_tensor:torch.Tensor, grid_dimension:tuple):
    '''
    Returns to original format.
    '''

    list_of_tensors = []


    for i in range(len(labels_tensor)):
        inner =[]
        x = labels_tensor[i,:,:]
        not_all_zero = x.any(dim=-1)
        for each in x[not_all_zero]:
            inner.append(each)
        if inner != []:
            list_of_tensors.append(inner)

    return list_of_tensors


def plot_detection_data(imgs, global_labels, start_idx=0):
    """Data should be global"""
    _, axes = plt.subplots(nrows=2, ncols=5, figsize=(8,3))

    for i, ax in enumerate(axes.flat): 
        
        img, labels = imgs[i+start_idx], global_labels[i+start_idx]
        img_height, img_width = img.shape[-2], img.shape[-1]
        img = img.clone()
        img = (img * 255).byte()
        labels = [labels] if not isinstance(labels, list) else labels
        label_classes = ''
        
        for label in labels:
            label_classes += f'{int(label[-1])}  '
            bbox = label[1:5]
            bbox = bbox.clone()
            bbox[0] *= img_width
            bbox[1] *= img_height
            bbox[2] *= img_width
            bbox[3] *= img_height

            bbox = bbox.type(torch.uint8)

            converted_bbox = box_convert(bbox, in_fmt='cxcywh', out_fmt='xyxy')

            img = draw_bounding_boxes(img, converted_bbox.unsqueeze(0), colors='lightgreen')

        img  = img.numpy().transpose((1, 2, 0))
        ax.imshow(img, cmap='gray')
        ax.set_title(label_classes)
        ax.axis('off')
        plt.suptitle(f'Image {start_idx} - {start_idx+9}')