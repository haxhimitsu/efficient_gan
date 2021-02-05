#!/usr/bin/env python3
# coding: UTF-8
import torch

class Anomaly_score:

    def __init__(self):
        self.Lambda=0.0

    def caloc_anomaly_score(self,x, fake_img, z_out_real, D, Lambda=0.1):
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
