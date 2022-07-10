from turtle import forward
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import sys
import torch
import torch.nn as nn
import glob

#コンフィグ
d_waves = 8
heads = 8
d_wave_width = 200
encoder_layers = 6

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
        self.pos_encoder = PositionalEncoding(d_waves)
        encoder_layer = nn.TransformerEncoderLayer(d_waves, heads, batch_first=True)
        self.transformerEncoder = nn.TransformerEncoder(encoder_layer, encoder_layers)
        self.flatten = nn.Flatten(1, 2)
        self.sequence = nn.Sequential(\
            nn.Linear(d_waves * (d_wave_width * 2 + 1), d_waves * d_wave_width),\
            nn.ReLU(),\
            nn.Linear(d_waves * (d_wave_width * 2 + 1), d_waves * d_wave_width),\
            nn.ReLU())
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.pos_encoder(x)
        x = self.transformerEncoder(x)
        x = self.flatten(x)
        x = self.sequence(x)
        out = self.softmax(x)
        return out

def data_make(self, filename:str):
    meta_df = pd.read_csv("../data/modified_meta.csv", encoding= "UTF-8")
    scratch_pos = meta_df.loc[filename, "scratchpos_all"]
    filepath = "../data/train/" + filename
    df = pd.read_csv(filepath, encoding= "UTF-8")
    length = len(df)
    data = []
    label = []
    for i in range(d_wave_width, length - 1 - d_wave_width):
        partial_df = df.iloc[i-d_wave_width, i+d_wave_width+1].to_numpy().transpose()
        data.append(torch.tensor(partial_df))
        before_points = 0
        for point in scratch_pos:
            if(point < i):
                before_points += 1
        if(before_points //2 == 1):
            label.append(torch.tensor([1, 0]))
        else:
            label.append(torch.tensor([0, 1]))
    return data, label

class DataSet(torch.utils.data.Dataset):
    def __init__(self) -> None:
        self.data = []
        self.label = []
        super().__init__()

    def __getitem__(self, idx):
        out_data = self.data[idx]
        out_label = self.label[idx]
        out_data = torch.tensor(out_data)
        return out_data, out_label



def Train():
    files = glob.glob("../data/train/*")
  
if __name__ == "__main__":
    arg = sys.argv
    if arg[1] == "__train":
        #訓練の挙動を定義
        files = glob.glob("../data/train/*")
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
