#!/usr/bin/env python3
# coding: UTF-8
#---------------------------------------------------------------
# author:"Haxhimitsu"
# date  :"2021/01/06"
# cite  :https://github.com/YutaroOgawa/pytorch_advanced/blob/master/6_gan_anomaly_detection/6-4_EfficientGAN.ipynb
# useage:python3 tf_sample_ver2.0.py  --train_path  ~/Desktop/dataset_smple/train/ --val_path ~/Desktop/dataset_smple/val/  --log_dir  ../test/ --test_data_path ~/Desktop/dataset_smple/test/
#---------------------------------------------------------------
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import time
import os
import csv
import copy
import random
import argparse
import sys
import matplotlib.pyplot as plt

#my create original module
from utils.data_loader import data_loader,GAN_Img_Dataset,ImageTransform
from utils.gan_model import Generator,Discriminator,Encoder
from train import train

data_load=data_loader()
#data_load.sayStr("Hello")
#test=train()
#test.sayStr("test")

# Setup seeds
torch.manual_seed(1234)
torch.cuda.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)


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

#create_data_path_list
print(val_path)
train_img_list,val_img_list=data_load.create_dataset(train_path=train_path,val_path=val_path)
#print("train_img_list\n",train_img_list)
print("val_img list\n",val_img_list)

#create_data_set
mean=(0.5,)
std=(0.5,)
train_dataset=GAN_Img_Dataset(
    file_list=train_img_list,transform=ImageTransform(mean,std)
)
test_dataset=GAN_Img_Dataset(
    file_list=val_img_list,transform=ImageTransform(mean,std)
)

#create_data_loader
batch_size=5
test_dataloader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=True)

batch_size = 64
train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True)


# show train image and test image
train_batch_iterator = iter(train_dataloader)
test_batch_iterator = iter(test_dataloader)  # イテレータに変換
train_images = next(train_batch_iterator)
test_images = next(test_batch_iterator)  # 1番目の要素を取り出す
#use matplot lib
fig = plt.figure(figsize=(15, 6))
## vew  test data
for i in range(0, 5):
    # upper columm is train data
    plt.subplot(2, 5, i+1)
    plt.imshow(train_images[i][0].cpu().detach().numpy(), 'gray')
    # under columm is test data
    plt.subplot(2, 5, 5+i+1)
    plt.imshow(test_images[i][0].cpu().detach().numpy(), 'gray')
    
plt.savefig("view_traindata_test_data"+".pdf",dpi=500)
#print(test_images.size())  # torch.Size([64, 1, 64, 64])

#init network weight
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        # conv2dとConvTranspose2dの初期化
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('BatchNorm') != -1:
        # BatchNorm2dの初期化
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('Linear') != -1:
        # 全結合層Linearの初期化
        m.bias.data.fill_(0)


#create_model
G = Generator(z_dim=20)
D = Discriminator(z_dim=20)
E = Encoder(z_dim=20)

#adapt init func
G.apply(weights_init)
E.apply(weights_init)
D.apply(weights_init)
print("init network weights!!")


# 学習・検証
num_epochs = 150
G_update, D_update, E_update = train.train_model(G, D, E, dataloader=train_dataloader, num_epochs=num_epochs)

##test


def Anomaly_score(x, fake_img, z_out_real, D, Lambda=0.1):

    # テスト画像xと生成画像fake_imgのピクセルレベルの差の絶対値を求めて、ミニバッチごとに和を求める
    residual_loss = torch.abs(x-fake_img)
    residual_loss = residual_loss.view(residual_loss.size()[0], -1)
    residual_loss = torch.sum(residual_loss, dim=1)

    # テスト画像xと生成画像fake_imgを識別器Dに入力し、特徴量マップを取り出す

    _, x_feature = D(x, z_out_real)
    _, G_feature = D(fake_img, z_out_real)

    # テスト画像xと生成画像fake_imgの特徴量の差の絶対値を求めて、ミニバッチごとに和を求める
    discrimination_loss = torch.abs(x_feature-G_feature)
    discrimination_loss = discrimination_loss.view(
        discrimination_loss.size()[0], -1)
    discrimination_loss = torch.sum(discrimination_loss, dim=1)

    # ミニバッチごとに2種類の損失を足し算する
    loss_each = (1-Lambda)*residual_loss + Lambda*discrimination_loss

    # ミニバッチ全部の損失を求める
    total_loss = torch.sum(loss_each)

    return total_loss, loss_each, residual_loss


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("使用デバイス：", device)

# 異常検知したい画像
x = test_images[0:5]
x = x.to(device)

# 教師データの画像をエンコードしてzにしてから、Gで生成
E_update.eval()
G_update.eval()
z_out_real = E_update(test_images.to(device))
imges_reconstract = G_update(z_out_real)

# 損失を求める
loss, loss_each, residual_loss_each = Anomaly_score(
    x, imges_reconstract, z_out_real, D_update, Lambda=0.1)

# 損失の計算。トータルの損失
loss_each = loss_each.cpu().detach().numpy()
print("total loss：", np.round(loss_each, 0))

# 画像を可視化
fig = plt.figure(figsize=(15, 6))
for i in range(0, 5):
    # 上段に訓練データを
    plt.subplot(3, 5, i+1)
    plt.imshow(train_images[i][0].cpu().detach().numpy(), 'gray')
    
    plt.subplot(3, 5, 5+i+1)
    plt.imshow(test_images[i][0].cpu().detach().numpy(), 'gray')
    
    # 下段に生成データを表示する
    plt.subplot(3, 5, 10+i+1)
    plt.imshow(imges_reconstract[i][0].cpu().detach().numpy(), 'gray')
    
plt.savefig("test_compare"+".png",dpi=500)