import sys
import pandas_datareader.data as data
import matplotlib.pyplot as plt
from torch import *

class StockPredictNet(nn.Module):
    def __init__(self, attnHead, hiddn, num_layers = 2,) -> None:
        

if __name__ == "__main__":
    arg = sys.argv
    if arg[1] == "__train":
        #訓練の挙動を定義
        pass
    if arg[1] == "__predict":
        #実行の挙動を定義
        pass