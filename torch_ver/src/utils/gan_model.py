#!/usr/bin/env python3
# coding: UTF-8
#---------------------------------------------------------------
# author:"Haxhimitsu"
# date  :"2021/01/06"
# cite  :
#---------------
import random
import math
import time
import pandas as pd
import numpy as np
from PIL import Image
# 動作確認
import matplotlib.pyplot as plt

import torch
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import transforms




#128pixel-32
#256pixel-64
class Generator(nn.Module):

    def __init__(self, z_dim=30):
        super(Generator, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(z_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True))

        self.layer2 = nn.Sequential(
            nn.Linear(1024, 32*32*128),
            nn.BatchNorm1d(32*32*128),
            nn.ReLU(inplace=True))

        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=64,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))

        self.last = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=1,
                               kernel_size=4, stride=2, padding=1),
            nn.Tanh())
        # 注意：出力チャネルは3つだけ

    def forward(self, z):
        out = self.layer1(z)
        out = self.layer2(out)

        # 転置畳み込み層に入れるためにテンソルの形を整形
        out = out.view(z.shape[0], 128, 32, 32)
        out = self.layer3(out)
        #print(out.size())
        out = self.last(out)
        #print(out.size())

        return out

class Discriminator(nn.Module):
    
    def __init__(self, z_dim=30):
        super(Discriminator, self).__init__()

        # 画像側の入力処理
        self.x_layer1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4,
                      stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True))
        # 注意：入力チャネルは3

        self.x_layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=4,
                      stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True))

        # 乱数側の入力処理
        self.z_layer1 = nn.Linear(z_dim,512)

        # 最後の判定 512+64*7*7=3648
        #self.last1 = nn.Sequential(nn.Linear(3648, 1024),nn.LeakyReLU(0.1, inplace=True))
        self.last1 = nn.Sequential(nn.Linear(66048, 1024),nn.LeakyReLU(0.1, inplace=True))
        self.last2 = nn.Linear(1024, 1)

    def forward(self, x, z):

        # 画像側の入力処理
        x_out = self.x_layer1(x)
        x_out = self.x_layer2(x_out)
        #print("layer2_passed_size",x_out.size())

        # 乱数側の入力処理
        z = z.view(z.shape[0], -1)
        #print("z",z.size())
        z_out = self.z_layer1(z)
        #print("z_out.size()",z_out.size())

        # x_outとz_outを結合し、全結合層で判定
        x_out = x_out.view(-1, 64 * 32 * 32)##change_imagesize
        #x_outs = x_out.view(-1, 64 * 7 * 7)
        #print("x_outs.size()",x_out.size())
        out = torch.cat([x_out, z_out], dim=1)
        out = self.last1(out)

        feature = out  # 最後にチャネルを1つに集約する手前の情報
        feature = feature.view(feature.size()[0], -1)  # 2次元に変換

        out = self.last2(out)

        return out, feature

class Encoder(nn.Module):
    
    def __init__(self, z_dim=30):
        super(Encoder, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3,
                      stride=1),
            nn.LeakyReLU(0.1, inplace=True))
        # 注意：白黒画像なので入力チャネルは1つだけ

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3,
                      stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True))

        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3,
                      stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True))

        # ここまでで画像のサイズは[128,128]to 32×32になっている
        self.last = nn.Linear(128 * 32 * 32, z_dim)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)

        # FCに入れるためにテンソルの形を整形
        out = out.view(-1, 128 * 32 * 32)##cahneg
        out = self.last(out)

        return out


if __name__ == '__main__':

    
    #generator check prog
    G = Generator(z_dim=20)
    G.train()
    # 入力する乱数
    # バッチノーマライゼーションがあるのでミニバッチ数は2以上
    input_z = torch.randn(64, 20)
    # 偽画像を出力
    fake_images = G(input_z)  # torch.Size([2, 1, 28, 28])
    img_transformed = fake_images[0][0].detach().numpy()
    plt.imshow(img_transformed, 'gray')
    #plt.show()

    #discriminator check
    D = Discriminator(z_dim=20)

    # 偽画像を生成
    #input_z = torch.randn(2, 20)
    print("fake_image_size",fake_images.size())

    # 偽画像をDに入力
    d_out, _ = D(fake_images, input_z)
    print("d_out.size()",d_out.size())
    # 出力d_outにSigmoidをかけて0から1に変換
    #print(nn.Sigmoid()(d_out))
    
    #Encoder check
    E = Encoder(z_dim=20)

    # 入力する画像データ
    x = fake_images  # fake_imagesは上のGで作成したもの
    print(fake_images.shape)
    # 画像からzをEncode
    z = E(x)

    print(z.shape)
    print(z)
    