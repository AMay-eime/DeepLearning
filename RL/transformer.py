"""
分類機を複数設定することによる強化学習。行動を直接出力するモデルになる。
エージェントの行動がたくさんある場合について強そう。近傍の行動が定義しやすいと尚よい。
環境の変数は複数のトークンで生成され、行動も同様にトークンで生成される。
モデルは以下の三つ。
[A]env_tokens -> action_tokens(encoder_decoder)
[B]env_tokens + action_tokens -> value(encoder)

1. playの際
Aモデルでアクション(a)を生成。そこから近傍のアクション(a_)を作成。
Bモデルで優劣を判断し焼きなましの要領で行動を決定していく。
（この際、学習の進行度に応じて作成する近傍アクションの遠さを操作しても良いか）
2.learningの際
Aモデルについては選択された真のaction_tokenとの誤差を縮めるように
BモデルについてはvalueをAdvantageに近づけるように。
"""

import numpy as np
import sys
import torch
import torch.nn as nn
import torch.optim as optim

#config
token_len = 20
env_token_num = 10
action_token_num = 2

encoder_layers = 6
decoder_layers = 6
epochs = 8
batch_size = 16
start_epoch = 0

gamma = 0.98
advantage_steps = 3

#便利な変換する奴らたち
def action_to_tokens(action):
    tokens = np.zeros(action_token_num, token_len)
    return tokens

def tokens_to_actions(tokens):
    action = np.zeros(4)
    return action

def env_to_tokens(env):
    tokens = np.zeros(env_token_num, token_len)
    return tokens

def tokens_to_env(tokens):
    env = np.zeros(5)
    return env

#近傍アクション生成機
def action_nearby(action, variance):
    a_ = action
    return a_

def action_nearby_token(token, variance):
    token_ = token
    return token_

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

class ActionNet(nn.Module):
    def __init__(self):
        super(ActionNet, self).__init__()
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

class ValueNet(nn.Module):
    def __init__(self):
        super(ActionNet, self).__init__()
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

def Play(value_net: ValueNet, action_net: ActionNet, variance):
    while False:#環境が終わるまで
        environment = 0
        a = action_net(environment)
        value = value_net(torch.concat([environment, a]))
        for i in range(10):#もちろん時間で区切ってもよし
            a_ = action_nearby_token(a)
            value_ = value_net(torch.concat([environment, a_]))
            if(value_ > value):
                a = a_
                value = value_
    return [], [], [] #各ステップのenv, act, returnの配列を吐き出させる

def Update(result, value_net:ValueNet):#マッチの結果をもとに学習をする。
    steps = len(result[1])
    for i in range(steps):
        s, s_, a = result[0][i], result[0][i+1], result[1][i]
        v = 0
        for j in range(i, min(steps, i + advantage_steps)):
            v += result[2][j] * pow(gamma, j-i)
        if (steps > i+advantage_steps):
            last_value = value_net(result[0][i + advantage_steps + 1])

def Train(result):
    for i in range(epochs):
        Play()

if __name__ == "__main__":
    arg = sys.argv
    if arg[1] == "__train":
        #訓練の挙動を定義
        Train()
    elif arg[1] == "__predict":
        #実行の挙動を定義
        pass