#!/usr/bin/env python3
# coding: UTF-8
#---------------------------------------------------------------
# author:"Haxhimitsu"
# date  :"2021/01/06"
# cite  :
# sample:python3 tf_sample_ver2.0.py  --train_path  ~/Desktop/dataset_smple/train/ --val_path ~/Desktop/dataset_smple/val/  --log_dir  ../test/ --test_data_path ~/Desktop/dataset_smple/test/

#---------------------------------------------------------------

import keras
from keras.utils import np_utils
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.preprocessing.image import array_to_img, img_to_array, load_img
import keras.callbacks
from keras.models import Sequential, model_from_json
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from tensorflow.keras.callbacks import EarlyStopping

import numpy as np
#import pandas as pd
#from sklearn.model_selection import train_test_split
#import matplotlib.pyplot as plt
import cv2
import os
import csv
import copy
import random
import argparse
import sys
#my module

from utils.data_loader import data_loader

data_load=data_loader()
data_load.sayStr("Hello")



parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path",required=True,help="path to root dataset directory")
parser.add_argument("--train_path",help="path to train_data")
parser.add_argument("--val_path",  help="set image size e.g.'0.5,0.8...'")
parser.add_argument("--max_epochs", type =int ,default=30,help="set trim width")
parser.add_argument("--save_weight_name", type=str,default="test",help="set trim height")
#parser.add_argument("--test_path",  help="output path")
parser.add_argument("--log_dir",  help="log_path")
a = parser.parse_args()

log_dir=a.log_dir
#myutil.create_directory(log_dir)
#weight_filename=a.save_weight_name+".hdf5"
#max_epochs=a.max_epochs

if a.train_path is None:
    train_path=a.dataset_path+"train/"
    #print("train_path",train_path)
else:
    train_path=a.train_path
    #print("train_path",train_path)
if a.val_path is None:
    val_path=a.dataset_path+"val/"
else:
    val_path=a.val_path


train_img_list,val_img_list=data_load.create_dataset(train_path,val_path)
#################setting GPU useage#####################
config = tf.ConfigProto(
    gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.9, # 最大値の80%まで
        allow_growth=True # True->必要になったら確保, False->全部
      ))
sess = sess = tf.Session(config=config)

#####################################################

