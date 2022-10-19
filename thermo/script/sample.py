from cProfile import label
from email import utils
from random import random
from turtle import forward
from typing import Tuple
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import glob
import math
import random

#コンフィグ
d_waves = 8
heads = 8
d_wave_width = 200
encoder_layers = 6
epochs = 8
batch_size = 16
restart_epoch = -1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class ScratchNet(nn.Module):
    def __init__(self):
        super(ScratchNet, self).__init__()
        self.pos_encoder = PositionalEncoding(d_waves)
        encoder_layer = nn.TransformerEncoderLayer(d_waves, heads, batch_first=True)
        self.transformerEncoder = nn.TransformerEncoder(encoder_layer, encoder_layers)
        self.flatten = nn.Flatten(1, 2)
        self.sequence = nn.Sequential(\
            nn.Linear(d_waves * (d_wave_width * 2 + 1), d_waves * d_wave_width),\
            nn.ReLU(),\
            nn.Linear(d_waves * d_wave_width, 2),\
            nn.ReLU())
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.pos_encoder(x)
        x = self.transformerEncoder(x)
        x = self.flatten(x)
        x = self.sequence(x)
        out = self.softmax(x)
        return out

def data_make(filename:str):
    meta_df = pd.read_csv("../data/modified_meta.csv", encoding= "UTF-8", sep = ",", index_col=0)
    scratch_pos:str = meta_df.at[filename, "scratchpos_all"]
    scratch_pos = scratch_pos.replace("[", "")
    scratch_pos = scratch_pos.replace("]", "")
    if(scratch_pos == ""):
        scratch_pos = []
    else:
        scratch_pos = scratch_pos.split(", ")
    #print(scratch_pos)
    filepath = "../data/train/" + filename
    df = pd.read_csv(filepath, encoding= "UTF-8", sep= ",")
    length = len(df)
    data = []
    label = []
    for i in range(d_wave_width, length - 1 - d_wave_width):
        partial_df = df[i-d_wave_width:i+d_wave_width+1].to_numpy().astype(np.float32) #(401,7)
        data.append(partial_df)
        before_points = 0
        for point in scratch_pos:
            if(int(point) < i):
                before_points += 1
        if(before_points //2 == 1):
            label.append(np.array([1., 0.]).astype(np.float32))
        else:
            label.append(np.array([0., 1.]).astype(np.float32))
    #print(f"data length = {len(data)}")
    return data, label

class DataSet(torch.utils.data.Dataset):
    def __init__(self) -> None:
        self.data = []
        self.label = []
        super().__init__()

    def add_items(self, datum, label):
        self.data.extend(datum)
        self.label.extend(label)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        out_data = self.data[idx]
        out_label = self.label[idx]
        out_data = torch.from_numpy(out_data)
        out_label = torch.from_numpy(out_label)
        out_data.to(device)
        out_label.to(device)
        return out_data, out_label

    def __len__(self) -> int:
        return len(self.data)

def Train():
    square_loss = nn.MSELoss()
    cross_entropy = nn.CrossEntropyLoss()
    net = ScratchNet()
    if restart_epoch > -1:
        net.load_state_dict(torch.load(f"model_{restart_epoch}.pth"))
    optimiser = optim.SGD(net.parameters(), lr=5e-4, momentum=0.9, nesterov= True)
    
    for i in range(epochs):
        files = glob.glob("../data/train/*")
        files = random.sample(files, 16)
        train_set = DataSet()
        for file in files:
            file = file.split("/")[-1]
            print(file)
            data, label = data_make(file)
            train_set.add_items(data, label)
        print(f"made train set epoch_{i}")

        running_loss = 0.0
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        for j, (inputs, labels) in enumerate(train_loader, 0):
            optimiser.zero_grad()

            outputs = net(inputs)
            loss = square_loss(outputs, labels)
            loss.backward()
            optimiser.step()
            running_loss += loss.item()
            if(j % 100 == 99):
                print(f"[{i+1}, {j+1}] loss: {running_loss/100}")
                running_loss = 0
        print(f"epoch finished {i}")
        torch.save(net.state_dict(), f"model_{i}.pth")
    print("finishing train")
  
if __name__ == "__main__":
    arg = sys.argv
    if arg[1] == "__train":
        #訓練の挙動を定義
        Train()
    elif arg[1] == "__predict":
        #実行の挙動を定義
        pass
    elif arg[1] == "__graph":
        data = pd.read_csv("../data/train/0a5a3a99.csv", encoding= "UTF-8")
        data_y_F1_X = data[data.columns[0]]
        data_y_F1_Y = data[data.columns[1]]
        data_y_F2_X = data[data.columns[2]]
        data_y_F2_Y = data[data.columns[3]]
        data_y_F3_X = data[data.columns[4]]
        data_y_F3_Y = data[data.columns[5]]
        data_y_MIX1_X = data[data.columns[6]]
        data_y_MIX1_Y = data[data.columns[7]]

        fig : Figure = plt.figure()

        ax1 = fig.add_subplot(2,4,1)
        ax2 = fig.add_subplot(2,4,2)
        ax3 = fig.add_subplot(2,4,3)
        ax4 = fig.add_subplot(2,4,4)
        ax5 = fig.add_subplot(2,4,5)
        ax6 = fig.add_subplot(2,4,6)
        ax7 = fig.add_subplot(2,4,7)
        ax8 = fig.add_subplot(2,4,8)

        ax1.plot(data_y_F1_X)
        ax2.plot(data_y_F1_Y)
        ax3.plot(data_y_F2_X)
        ax4.plot(data_y_F2_Y)
        ax5.plot(data_y_F3_X)
        ax6.plot(data_y_F3_Y)
        ax7.plot(data_y_MIX1_X)
        ax8.plot(data_y_MIX1_Y)

        plt.show()
        #見たところ総長401くらいでイケそう、傷のマークが左川に奇数個あればpositiveラベル。50個区間の(positive num)/50を出力で。
        #また、極端に近いところや遠いところはラベルされていなさそう。
    elif arg[1] == "__test_construct":
        df_meta = pd.read_csv("../data/train_meta.csv", encoding= "UTF-8")
        df_meta["scratchpos_edge"] = df_meta["scratchpos_edge"].apply( lambda x: [int(y) for y in x.split()] if not pd.isna(x) else [])
        df_meta["scratchpos_center"] = df_meta["scratchpos_center"].apply( lambda x: [int(y) for y in x.split()] if not pd.isna(x) else [])
        df_meta["scratchpos_baffle"] = df_meta["scratchpos_baffle"].apply( lambda x: [int(y) for y in x.split()] if not pd.isna(x) else [])

        df_meta["num_edge"] = df_meta["scratchpos_edge"].apply( lambda x: int(len(x) / 2))
        df_meta["num_center"] = df_meta["scratchpos_center"].apply( lambda x: int(len(x) / 2))
        df_meta["num_baffle"] = df_meta["scratchpos_baffle"].apply( lambda x: int(len(x) / 2))

        df_meta["num_all"] = df_meta["num_edge"] + df_meta["num_center"] + df_meta["num_baffle"]

        def merge_scratch(row):
            
            merged_pos = row["scratchpos_edge"] + row["scratchpos_center"] + row["scratchpos_baffle"]
            merged_pos.sort()

            return merged_pos

        df_meta["scratchpos_all"] = df_meta.apply(merge_scratch, axis=1)
        df_meta.to_csv("../data/modified_meta.csv", index= False, columns=["filename", "scratchpos_edge", "scratchpos_center", "scratchpos_baffle", "scratchpos_all"])
