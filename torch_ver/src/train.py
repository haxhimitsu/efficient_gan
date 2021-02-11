#!/usr/bin/env python3
# coding: UTF-8
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


import time


class train:

    def __init__(self):
                # 最適化手法の設定
        self.lr_ge = 0.0001
        self.lr_d = 0.0001/4
        self.beta1, self.beta2 = 0.5, 0.999

    def train_model(self,G,D,E,dataloader,num_epochs):
        
        # GPUが使えるかを確認
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("使用デバイス：", device)

        g_optimizer = torch.optim.Adam(G.parameters(), self.lr_ge, [self.beta1,self.beta2])
        e_optimizer = torch.optim.Adam(E.parameters(), self.lr_ge, [self.beta1, self.beta2])
        d_optimizer = torch.optim.Adam(D.parameters(), self.lr_d, [self.beta1,self.beta2])

        # 誤差関数を定義
        # BCEWithLogitsLossは入力にシグモイド（logit）をかけてから、
        # バイナリークロスエントロピーを計算
        criterion = nn.BCEWithLogitsLoss(reduction='mean')

        # パラメータをハードコーディング
        z_dim = 30
        mini_batch_size = 64

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
                #print("mini_batch_size",mini_batch_size)
                label_real = torch.full((mini_batch_size,), fill_value=1.0).to(device)
                #print("label_real.size()",label_real.size())
                #print("label_real",label_real)
                label_fake = torch.full((mini_batch_size,), fill_value=0.0).to(device)

                # GPUが使えるならGPUにデータを送る
                imges = imges.to(device)

                # --------------------
                # 1. Discriminatorの学習
                # --------------------
                # 真の画像を判定　
                z_out_real = E(imges)
                #print("image",imges.size())
                #print("z_out_real.size",z_out_real.size())
                #print("z_out_real.shape()",z_out_real.shape)
                d_out_real, _ = D(imges, z_out_real)
                #print("d_out_real",d_out_real.size())
                #print("d_out_real.vew(-1)",d_out_real.view(-1).size())

                # 偽の画像を生成して判定
                input_z = torch.randn(mini_batch_size, z_dim).to(device)
                fake_images = G(input_z)
                d_out_fake, _ = D(fake_images, input_z)
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

    def sayStr(self, str):
        print (str)
