# Project2_265
Project 2 INF265 @uib

## Description

In this project we define and train convolutional neural networks to solve an object localization
task and an object detection task. In the object localization task there is at most one multiclass object in the image.
The object detection task allows for more than one image. The project also contains our own implementation of loss functions
and performance metrics (Mean of IOU and Accuracy for object localization and mAP for object detection.)

## Requirements and Setup

A basic PyTorch environment is needed.
Dependencies can be installed using requirements.txt.

File structure: 

├── data  
│   ├── detection_test.pt
│   ├── detection_train.pt
│   ├── detection_val.pt
│   ├── list_y_true_train.pt
│   ├── list_y_true_val.pt
│   ├── list_y_true_test.pt
│   ├── localization_test.pt
│   ├── localization_val.pt
│   └── localization_train.pt
├── test_results  # folder where data are saved 
├── project2.ipynb  # main file
├── project2_constants.py
├── project2_functions.py  
├── project2_objects.py
├── project2_models.py
├── requirements.txt
└── report.md

## Contact

- ain015@uib.no
- antok8704@uib.no
