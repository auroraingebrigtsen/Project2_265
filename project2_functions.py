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

from project2_objects import Timer
from project2_constants import DEVICE as device
from project2_constants import SEED


def count_instances(data:torch.Tensor, data_name=None) -> None:
    """
    Counts the number of instances of each class in a object localization dataset.
    If P_c is 0, counts the class as 99
    """
    counter = Counter([99 if label[0] == 0 else int(label[-1]) for _, label in data])
    sorted_counter = dict(sorted(counter.items()))
    if data_name is not None:
        print(f'Class distribution in {data_name}')
    for key, value in sorted_counter.items():
        print(f'{key}: {value}') 


def plot_localization_data(imgs:torch.Tensor, y_true:torch.Tensor, y_preds:torch.Tensor=None, class_label:int=None, start_idx:int=0, save_dir:str='test_results/', fig_name='new', save_model:bool=False) -> None:
    """
    Function used for plotting in object localization.
    If class_label is None, plots the first image of each class in the dataset starting at start_idx, with predictions if given. 
    If class_label is spesificed, plots a subplot with 10 images from a given class, starting at a chosen index, with predictions if given
    Saves results if save_model is True
    """
    
    plot_predictions = y_preds is not None
    class_specified = class_label is not None
    
    imgs = imgs.cpu()
    y_true = y_true.cpu()
    if plot_predictions:
        y_preds = y_preds.cpu()

    if class_specified:
        class_mask = (y_true[:, -1] == class_label) & (y_true[:, 0] == 1)
        class_imgs = imgs=imgs[class_mask]
        class_labels = y_true[class_mask]

        if plot_predictions:
            class_preds =  y_preds[class_mask]

    _, axes = plt.subplots(nrows=2, ncols=6, figsize=(8,3))

    for i, ax in enumerate(axes.flat): 

        if i == 10 and not class_specified and not plot_predictions:  # plot an image of empty img as well
            img = next(img for img, label in zip(imgs, y_true) if int(label[0]) == 0)
            img = img.numpy().transpose((1, 2, 0))
            ax.imshow(img, cmap='gray')
            ax.set_title('None')
            ax.axis('off')
            continue

        elif i == 11 and not class_specified and not plot_predictions:  # no more classes to plot
            ax.axis('off')
            continue
        
        elif not class_specified and not plot_predictions:  # plot one image of each class
            next_index = next((index for index, label in enumerate(y_true) if int(label[-1]) == i), None)
            img = imgs[next_index]
            label = y_true[next_index]
            ax.set_title(i)
        
        elif not class_specified and plot_predictions:  # plot 10 first prediction from start_idx
            img, label, y_pred =  imgs[start_idx+i], y_true[start_idx+i], y_preds[start_idx+i]
            ax.set_title(f'T:{int(label[-1]) if label[0] == 1 else None} P:{int(y_pred[-1]) if y_pred[0] > 0.5 else None}')

        else:
            img, label = class_imgs[start_idx+i], class_labels[start_idx+i]  # plot 10 first imgs from the given class, from start_idx
            plt.suptitle(f'Class {class_label} - Image {start_idx} - {start_idx+i}')

            if plot_predictions: # and also the prediction
                y_pred = class_preds[start_idx+i]
                ax.set_title(f'P:{int(y_pred[-1]) if y_pred[0] > 0.5 else None}')

        img_height, img_width = img.shape[-2], img.shape[-1]
        
        img = img.clone()
        img = (img * 255).byte()

        if label[0] == 1:  # plot true bounding box
            converted_bbox = _convert_box(label, img_width, img_height)
            img = draw_bounding_boxes(img, converted_bbox, colors='lightgreen')

        if plot_predictions:  # plot predicted bounding box
            if F.sigmoid(y_pred[0]) > 0.5:
                pred_bbox_converted = _convert_box(y_pred, img_width, img_height)
                img = draw_bounding_boxes(img, pred_bbox_converted, colors='red')

        img  = img.numpy().transpose((1, 2, 0))
        ax.imshow(img, cmap='gray')
        ax.axis('off')
        
    if save_model:
        plt.savefig(save_dir+fig_name+'_bbox_pred', bbox_inches='tight')


def fc_size(size:tuple, layers) -> int:
    """
    Calculates the input size of the first FC layer, given the input size of the image, and the layer objects in the model
    """
    input_size = size

    for layer in layers:
        if isinstance(layer, torch.nn.modules.linear.Linear):
            break
        
        elif isinstance(layer, torch.nn.modules.conv.Conv2d) or isinstance(layer, torch.nn.modules.pooling.MaxPool2d):
            input_size = get_output_size(input_size ,layer)     
        
    network_size = input_size[0] * input_size[1] * input_size[2]
    return int(network_size)
        

def get_output_size(input_size:tuple, layer:nn.Module) -> tuple:
    """
    Computes the output size of a given layer. Input size must be specified.
    """
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


def int_to_pair(n) -> tuple:
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
    

def localization_performance(model, loader) -> list:
    '''
    Calculates accuracy and IOU. Returns Accuracy, IOU and the mean of IOU and Accuracy respectively.
    '''
    model.eval()
    correct = 0
    total = 0
    iou_sum = 0
    total_iou = 0

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
            total_iou += outputs[object_detected].shape[0]
            
            correct += ((object_detected == 0) & (det_true == 0)).sum()
            correct += ((object_detected == 1) & (det_true == 1) & (class_pred == class_true)).sum()

            iou_sum += calculate_iou(outputs[object_detected], labels[object_detected]).sum().item()
            
    acc =  correct / total
    iou = iou_sum / total_iou

    performance = (acc + iou) / 2
    
    return [acc.item(), iou, performance.item()]


def plot_loss(train_loss:list, val_loss:list, title:str, save_dir='test_results/', save_model=False) -> None:
    """
    Plots the training and validation loss over epochs
    """
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


def plot_lists(data, loss_names:list, title:str, save_dir='test_results/', save_model=False) -> None:
    """
    Plots losses of seperate loss functions over epochs
    """
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
    

def train(n_epochs:int, optimizer, model, loss_fn, train_loader, val_loader, performance_calculator):
    """
    Trains a model, and also prints training and validation loss for each epoch. Computes training and validation performance. 
    Performance calculator should be a function to compute performance. The performance should be returned from the function in a list
    with the main metric as pos -1. 
    Returns training loss, validation loss, training performance, validation performance and the seperate losses L_a, L_b and L_c) trough epochs, respectively.
    """
    
    n_batch_train = len(train_loader)
    n_batch_val = len(val_loader)

    losses_train = []
    losses_val = []
    losses_separated = []
    
    time_spent = 0

    optimizer.zero_grad(set_to_none=True)
        
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

                loss,_ = loss_fn(outputs, labels)
                loss_val += loss.item()
            
        losses_train.append(loss_train / n_batch_train)
        losses_val.append(loss_val / n_batch_val)
        losses_separated.append([each/n_batch_train for each in inner_losses_separated])
        
        # Calculates time left of training
        time_spent += timer.stop_timer()
        time_left = (time_spent/epoch)*(n_epochs-epoch)
                
        print('{}  |  Epoch {}  |  Training loss {:.3f}'.format(datetime.now().strftime('%H:%M:%S'), epoch, loss_train / n_batch_train))
        print('{}  |  Epoch {}  |  Validation loss {:.3f}'.format(datetime.now().strftime('%H:%M:%S'), epoch, loss_val / n_batch_val))
        print(f"Estimated time left: {floor(time_left/60)}m {round(time_left%60)}s")
    
    train_performance = performance_calculator(model, train_loader)
    val_performance = performance_calculator(model, val_loader)
    print(f'Training performance: {train_performance}')
    print(f'Validation performance:{val_performance}')

    return losses_train, losses_val, train_performance, val_performance, losses_separated
    

def model_selector(models:list, performances:list):
    """
    Given a list of models, returns the model that has best accuracy score on validation data.
    Performances should be a list with main metric as pos -1.
    """
    best_model = None
    best_performance = 0

    for idx, model in enumerate(models):
        if performances[idx][-1] > best_performance:
            best_model = model
            best_performance = performances[idx][-1]

    return best_model, best_performance


def predict(model, loader, binary_class = False):
    '''
    Makes predictions given a model and a loader.
    If binary_class is True, applies sigmoid to the pos -1 at each output, and sets class 1 if 
    > 0.5 and 0 if not. 
    '''
    model.eval()
    
    
    y_true = torch.empty(0, device=device)
    y_pred = torch.empty(0, device=device)

    with torch.inference_mode(): # <-- Equivalent to no_grad, if no error is provided this is preferred.
        for imgs, labels in loader:
            
            imgs = imgs.to(device=device, dtype=torch.double) 
            labels = labels.to(device=device)
            outputs = model(imgs)
                        
            if binary_class:
                class_pred = torch.sigmoid(outputs[:, -1])
                class_pred = torch.where(class_pred>0.5, 1, 0)
            else: 
                _, class_pred = torch.max(outputs[:, 5:], dim=1)

            predicted = torch.cat((outputs[:, :5], class_pred.unsqueeze(1)), dim=1)
            
            y_true = torch.cat((y_true, labels), dim=0)
            y_pred = torch.cat((y_pred, predicted.data), dim=0)
                
    return y_true, y_pred


def global_to_local(labels_list:list, grid_dimension:tuple) -> torch.Tensor:
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

def prepare_labels(label_dataset:list, grid_dimension:tuple) -> torch.Tensor:
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



def plot_detection_data(imgs:torch.Tensor, y_true, y_pred=None, start_idx:int=0, save_dir:str='test_results/', fig_name:str='new', save_model:bool=False) -> None:
    """
    Plots detection data. Expects y_true and y_pred to be global.
    If y_pred is not specified, plots the true labels starting from index 0. 
    If y_pred is specified, plots the corresponding predictions as well.
    """
    _, axes = plt.subplots(nrows=2, ncols=5, figsize=(8,3))

    for i, ax in enumerate(axes.flat): 
        
        img, y_true_label = imgs[i+start_idx], y_true[i+start_idx]
        img_height, img_width = img.shape[-2], img.shape[-1]
        img = img.clone()
        img = (img * 255).byte()
        y_true_label = [y_true_label] if not isinstance(y_true_label, list) else y_true_label
        label_classes = ''
        
        for label in y_true_label:
            label = label.clone()
            label_classes += f'T: {int(label[-1])} '
            converted_bbox = _convert_box(label, img_width, img_height)
            img = draw_bounding_boxes(img, converted_bbox, colors='lightgreen')

        if y_pred is not None:
            label_classes += f'\n'
            y_pred_label = y_pred[i+start_idx]
            y_pred_label = [y_pred_label] if not isinstance(y_pred_label, list) else y_pred_label
            for label in y_pred_label:
                label = label.clone()
                label_classes += f'P: {int(label[-1])} '
                converted_bbox = _convert_box(label, img_width, img_height)
                img = draw_bounding_boxes(img, converted_bbox, colors='red')


        img  = img.numpy().transpose((1, 2, 0))
        ax.imshow(img, cmap='gray')
        ax.set_title(label_classes)
        ax.axis('off')

    if save_model:
        plt.savefig(save_dir+fig_name+'_bbox_pred', bbox_inches='tight')


def _convert_box(label:torch.Tensor, w:int, h:int) -> torch.Tensor:
    """
    Used to slice out the bbox from a label. Scales the bbox according to image width and image heigth.
    Uses pytorch's function box_convert to change format of tensor
    """
    bbox = label[1:5]
    bbox = bbox.clone()
    bbox[0] *= w
    bbox[1] *= h
    bbox[2] *= w
    bbox[3] *= h

    bbox = bbox.type(torch.uint8)

    converted_bbox = box_convert(bbox, in_fmt='cxcywh', out_fmt='xyxy')
    converted_bbox = converted_bbox.unsqueeze(0)
    return converted_bbox


def calculate_ap(outputs_reshaped, labels_reshaped, treshold:float = 0.5):
    """
    Calculates Average Precision.
    Treshold is the limit that decides wheter a tensor is a TP or FP. 
    TP = IOU ≥ threshold, FP = IOU < threshold
    """
    confidence = F.sigmoid(outputs_reshaped[:, 0])  # calculate confidences
    iou = calculate_iou(outputs_reshaped, labels_reshaped)  # calculate IOU
    tp = torch.where(iou >= treshold, 1, 0)
    fp = torch.where(iou < treshold, 1, 0)

    _, indices = torch.sort(confidence, dim = 0, descending=True) 
    ground_truths = (labels_reshaped[:,0] == 1).sum().item()

    tensor_length = len(indices)

    recall = torch.zeros(tensor_length)
    precision = torch.zeros(tensor_length)
    acc_tp = torch.zeros(tensor_length)
    acc_fp = torch.zeros(tensor_length)

    counter = 0

    # calculate recall and precision accumulated, starting from the prediction with highest confidence
    for i in indices:
        if counter == 0:
            acc_tp[counter] = tp[i]
            acc_fp[counter] = fp[i]

        else:
            acc_tp[counter] = tp[i]+acc_tp[counter-1]
            acc_fp[counter] = fp[i]+acc_fp[counter-1]

        precision[counter] = acc_tp[counter]/(acc_tp[counter]+acc_fp[counter])

        recall[counter] = acc_tp[counter]/ground_truths

        counter += 1

    # 11-point interpolation    
    interpolated_sum = 0

    recall_levels = torch.arange(0, 1.1, 0.1)

    for recall_level in recall_levels:
        mask = recall < recall_level
        if torch.any(mask):
            interpolated_sum += torch.max(precision[mask])

    return interpolated_sum / len(recall_levels)


def detection_performance(model, loader) -> list:
    '''
    Calculates detection performance, mAP, which is the mean of AP across classes. 
    '''
    model.eval()
    map_sum = 0
    total = 0
    with torch.inference_mode():
        for imgs, labels in loader:
            imgs = imgs.to(device=device, dtype=torch.double)
            labels = labels.to(device=device, dtype=torch.double)

            outputs = model(imgs)
            
            y_pred = outputs.permute(0,2,3,1)
            y_true = labels.permute(0,2,3,1)
            
            y_pred_reshaped = y_pred.reshape(-1, y_pred.size(-1))
            y_true_reshaped = y_true.reshape(-1, y_true.size(-1))

            classes = torch.unique(y_true_reshaped[:,-1])
            
            ap_sum = 0

            for each in classes:
                mask = y_true_reshaped[:,-1] == each

                ap_sum += calculate_ap(y_pred_reshaped[mask], y_true_reshaped[mask])
                
   
            map_sum += ap_sum/len(classes)
            total += 1
            
    return [map_sum/total]


def calculate_iou(outputs, labels):
    """
    Calculate IoU between ground truth and predicted boxes.
    """

    bbox_pred = outputs[:, 1:5].clone()
    bbox_true = labels[:, 1:5].clone()

    converted_bbox_pred = box_convert(bbox_pred, in_fmt='cxcywh', out_fmt='xyxy')
    converted_bbox_true = box_convert(bbox_true, in_fmt='cxcywh', out_fmt='xyxy')

    bbox_iou = box_iou(converted_bbox_pred,converted_bbox_true)
    
    iou = bbox_iou.diag()  # selects only diagonal IOU's, since bbox_iou returns  pairwise IoU values for every element in boxes1 and boxes2
    
    return iou

def local_to_global_list(input_tensor:torch.Tensor):
    """
    Changes a local tensor back to the original listed format.
    """

    input_tensor = input_tensor.clone()

    returned_list = []

    h_size = input_tensor.shape[1]
    w_size = input_tensor.shape[2]

    for h in range(h_size):


        for w in range(w_size):

            input_tensor[:,h,w,1] /=3
            input_tensor[:,h,w,2] /=2

            input_tensor[:,h,w,1] += (w*1)/w_size
            input_tensor[:,h,w,2] += (h*1)/h_size

            input_tensor[:,h,w,3] /=3
            input_tensor[:,h,w,4] /=2

    new_tensor = input_tensor.view(-1, input_tensor.size(-1), input_tensor.size(-1))
    new_tensor_sigmoid = torch.sigmoid(new_tensor[:, :, 0])
    mask = new_tensor_sigmoid > 0.5
    filtered_tensors = [new_tensor[i][mask[i]] for i in range(new_tensor.size(0))]

    for each in filtered_tensors:
        inner = []
        for i in each:
            inner.append(i)
        returned_list.append(inner)

    return returned_list


def normalizer(source_dataset, val_dataset, test_dataset):
    '''
    A function to normalize the data based on mean and standard.
    Source dataset is the dataset to normalize from.
    '''
    imgs = torch.stack([img for img, _ in source_dataset])

    # Define normalizer
    normalizer_pipe = transforms.Normalize(
        imgs.mean(dim=(0, 2, 3)), 
        imgs.std(dim=(0, 2, 3))
        )

    # Define preprocessor including the normalizer
    preprocessor = transforms.Compose([
                normalizer_pipe
            ])
    
    source_dataset_norm = [(preprocessor(img), label) for img, label in source_dataset]
    val_dataset_norm = [(preprocessor(img), label) for img, label in val_dataset]
    test_dataset_norm = [(preprocessor(img), label) for img, label in test_dataset]
    
    return source_dataset_norm, val_dataset_norm, test_dataset_norm
