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
#my module

from utils.data_loader import data_loader,GAN_Img_Dataset,ImageTransform
from utils.gan_model import Generator,Discriminator,Encoder
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

#create_data_path_list
train_img_list,val_img_list=data_load.create_dataset(train_path,val_path)
#print("train_img_list\n",train_img_list)
#print("val_img list\n",train_img_list)

#create_data_set
mean=(0.5,)
std=(0.5,)
train_dataset=GAN_Img_Dataset(
    file_list=train_img_list,transform=ImageTransform(mean,std)
)
#create_data_loader
batch_size = 64
train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True)


# 動作の確認
batch_iterator = iter(train_dataloader)  # イテレータに変換
imges = next(batch_iterator)  # 1番目の要素を取り出す
print(imges.size())  # torch.Size([64, 1, 64, 64])


#create_model
G = Generator(z_dim=20)
D = Discriminator(z_dim=20)
E = Encoder(z_dim=20)

def train_model(G,D,E,dataloader,num_epochs):

    # GPUが使えるかを確認
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("使用デバイス：", device)

    # 最適化手法の設定
    lr_ge = 0.0001
    lr_d = 0.0001/4
    beta1, beta2 = 0.5, 0.999
    g_optimizer = torch.optim.Adam(G.parameters(), lr_ge, [beta1, beta2])
    e_optimizer = torch.optim.Adam(E.parameters(), lr_ge, [beta1, beta2])
    d_optimizer = torch.optim.Adam(D.parameters(), lr_d, [beta1, beta2])

    # 誤差関数を定義
    # BCEWithLogitsLossは入力にシグモイド（logit）をかけてから、
    # バイナリークロスエントロピーを計算
    criterion = nn.BCEWithLogitsLoss(reduction='mean')

    # パラメータをハードコーディング
    z_dim = 20
    mini_batch_size = 4

    # ネットワークをGPUへ
    G.to(device)
    E.to(device)
    D.to(device)

    G.train()  # モデルを訓練モードに
    E.train()  # モデルを訓練モードに
    D.train()  # モデルを訓練モードに

    # ネットワークがある程度固定であれば、高速化させる
    torch.backends.cudnn.benchmark = True

    # 画像の枚数
    num_train_imgs = len(dataloader.dataset)
    batch_size = dataloader.batch_size

    # イテレーションカウンタをセット
    iteration = 1
    logs = []

    # epochのループ
    for epoch in range(num_epochs):

        # 開始時刻を保存
        t_epoch_start = time.time()
        epoch_g_loss = 0.0  # epochの損失和
        epoch_e_loss = 0.0  # epochの損失和
        epoch_d_loss = 0.0  # epochの損失和

        print('-------------')
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-------------')
        print('（train）')

        # データローダーからminibatchずつ取り出すループ
        for imges in dataloader:

            # ミニバッチがサイズが1だと、バッチノーマライゼーションでエラーになるのでさける
            if imges.size()[0] == 1:
                continue

            # ミニバッチサイズの1もしくは0のラベル役のテンソルを作成
            # 正解ラベルと偽ラベルを作成
            # epochの最後のイテレーションはミニバッチの数が少なくなる
            mini_batch_size = imges.size()[0]
            print("mini_batch_size",mini_batch_size)
            label_real = torch.full((mini_batch_size,), fill_value=1.0).to(device)
            print(label_real)
            label_fake = torch.full((mini_batch_size,), fill_value=0.0).to(device)

            # GPUが使えるならGPUにデータを送る
            imges = imges.to(device)

            # --------------------
            # 1. Discriminatorの学習
            # --------------------
            # 真の画像を判定　
            z_out_real = E(imges)
            d_out_real, _ = D(imges, z_out_real)

            # 偽の画像を生成して判定
            input_z = torch.randn(mini_batch_size, z_dim).to(device)
            fake_images = G(input_z)
            d_out_fake, _ = D(fake_images, input_z)
            print("hoge")
            # 誤差を計算
            d_loss_real = criterion(d_out_real.view(-1), label_real)
            d_loss_fake = criterion(d_out_fake.view(-1), label_fake)
            d_loss = d_loss_real + d_loss_fake

            # バックプロパゲーション
            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            # --------------------
            # 2. Generatorの学習
            # --------------------
            # 偽の画像を生成して判定
            input_z = torch.randn(mini_batch_size, z_dim).to(device)
            fake_images = G(input_z)
            d_out_fake, _ = D(fake_images, input_z)

            # 誤差を計算
            g_loss = criterion(d_out_fake.view(-1), label_real)

            # バックプロパゲーション
            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            # --------------------
            # 3. Encoderの学習
            # --------------------
            # 真の画像のzを推定
            z_out_real = E(imges)
            d_out_real, _ = D(imges, z_out_real)

            # 誤差を計算
            e_loss = criterion(d_out_real.view(-1), label_fake)

            # バックプロパゲーション
            e_optimizer.zero_grad()
            e_loss.backward()
            e_optimizer.step()

            # --------------------
            # 4. 記録
            # --------------------
            epoch_d_loss += d_loss.item()
            epoch_g_loss += g_loss.item()
            epoch_e_loss += e_loss.item()
            iteration += 1

        # epochのphaseごとのlossと正解率
        t_epoch_finish = time.time()
        print('-------------')
        print('epoch {} || Epoch_D_Loss:{:.4f} ||Epoch_G_Loss:{:.4f} ||Epoch_E_Loss:{:.4f}'.format(
            epoch, epoch_d_loss/batch_size, epoch_g_loss/batch_size, epoch_e_loss/batch_size))
        print('timer:  {:.4f} sec.'.format(t_epoch_finish - t_epoch_start))
        t_epoch_start = time.time()

    print("総イテレーション回数:", iteration)

    return G, D, E

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


#adapt init func
G.apply(weights_init)
E.apply(weights_init)
D.apply(weights_init)
print("init network weights!!")


# 学習・検証を実行する
# 15分ほどかかる
num_epochs = 1500
G_update, D_update, E_update = train_model(
    G, D, E, dataloader=train_dataloader, num_epochs=num_epochs)