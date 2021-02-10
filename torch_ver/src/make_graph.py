#!/usr/bin/env python3
# coding: UTF-8

import os
import csv
import copy
import random
import argparse
import sys
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument("--csv_path",required=True,help="path to root dataset directory")
# parser.add_argument("--train_path",help="path to train_data")
# parser.add_argument("--val_path",  help="set image size e.g.'0.5,0.8...'")
# parser.add_argument("--max_epochs", type =int ,default=500,help="set trim width")
# parser.add_argument("--save_weight_name", type=str,default="test",help="set trim height")
# parser.add_argument("--batch_size", default=64, type=int,help="output path")
# parser.add_argument("--log_dir",  default="./log/",help="log_path")
# parser.add_argument("--mode",  type=str,required=True,default="train",help="log_path")
# parser.add_argument("--result_dir", type=str,default="./img_result/",help="log_path")
a = parser.parse_args()

print(a.csv_path)

csv_file = open(a.csv_path, "r", encoding="ms932", errors="", newline="" )
#リスト形式
f = csv.reader(csv_file, delimiter=",", doublequote=True, lineterminator="\r\n", quotechar='"', skipinitialspace=True)
header = (f)
print(header)

data_class=[]
result =[]
for row in f:
    #rowはList
    #row[0]で必要な項目を取得することができる
    data_class.append(row[0])
    result.append(row[2])

print(data_class)
print(result)

fig, ax = plt.subplots()
plt.bar(data_class, result, color="#FF5B70", edgecolor="#CC4959", linewidth=0,align="center")
plt.title("Each dataset a1 score trained label03")
fig.subplots_adjust(left=0.2)
plt.xlabel("data label")
plt.ylabel("a1 score")
plt.grid(True)
#plt.show()

savepath=os.path.dirname(a.csv_path)
print(savepath)
plt.savefig(savepath+"/"+"view a1 score"+".pdf",dpi=800)