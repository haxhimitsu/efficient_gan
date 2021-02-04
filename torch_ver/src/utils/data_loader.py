#!/usr/bin/env python3
# coding: UTF-8

#---------------------------------------------------------------
# author:"Haxhimitsu"
# date  :"2021/01/06"
# cite  :
# sample:python3 imgtrim_gui_ver.2.0.py  --input_dir ../assets/original_img/cbn_test_01/ --output_dir ../assets/sample_output/  --trim_width 32 --trim_height 64
#---------------------------------------------------------------
import numpy as np
#import pandas as pd
#from sklearn.model_selection import train_test_split
#import matplotlib.pyplot as plt
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms

import os
from PIL import Image

class data_loader:

    def create_dataset(self,train_path,val_path):

        TrainIMG = []
        TrainLABEL = []
        ValIMG = []
        ValLABEL = []
        TestIMG = []
        TestLABEL = []
        label =0

        img_dirs=os.listdir(train_path)

        for i, d in enumerate(img_dirs):
            files0 = os.listdir(train_path+ d)
            files1 = os.listdir(val_path+ d)
            print(train_path+d)
            print(val_path+d)
            for f0 in files0:
                train_img_path=train_path + d + '/' + f0
                TrainIMG.append(train_img_path)

            for f1 in files1:
                val_img_path=val_path + d + '/' + f1
                ValIMG.append(val_img_path)

            print("now:" + img_dirs[i])
            #print("train_img_list\n",TrainIMG)
            #print("val_img list\n",ValIMG)

        return TrainIMG,ValIMG
    
    def sayStr(self, str):
        print (str)


class ImageTransform():
    """画像の前処理クラス"""

    def __init__(self, mean, std):
        self.data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    def __call__(self, img):
        return self.data_transform(img)

class GAN_Img_Dataset(data.Dataset):
    """画像のDatasetクラス。PyTorchのDatasetクラスを継承"""

    def __init__(self, file_list, transform):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        '''画像の枚数を返す'''
        return len(self.file_list)

    def __getitem__(self, index):
        '''前処理をした画像のTensor形式のデータを取得'''

        img_path = self.file_list[index]
        img = Image.open(img_path)  # [高さ][幅]白黒

        # 画像の前処理
        img_transformed = self.transform(img)

        return img_transformed

if __name__ == '__main__':
    test = data_loader()
    test.sayStr("Hello")   # Hello