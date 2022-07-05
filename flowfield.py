'''
유동장 데이터 클래스
'''
import pandas as pd
import numpy as np
import torch
import PIL
class Csvdata():
    def __init__(self, csv_dir):
        self.csv_dir = csv_dir
        self.df_data = pd.read_csv(self.csv_dir)
        self.data_factor = self.df_data[0,:]
        self.shape = self.

    def __getitem__(self):
        return self.img_dir

    def __len__(self):
        return self.df_data.shape[0]

class Flowfield():
    def __init__(self, field_csv, label_csv):
        self.label_dir = label_csv
        self.field_dir = field_csv
        self.label_data = pd.read_csv(self.label_dir)
        self.field_data = pd.read_csv(self.field_dir)
        self.data = [field_csv, label_csv]
        self.num_=0
        self.num_design=0
    
    def __getitem__(self):
        return self.data
    
    def __len__(self):
        return self.num_ 

    def astype(self, dtype):
        if dtype == 'numpy':
            data_ = self.data
        elif dtype == 'img':
            data_ = self.data
        elif dtype == 'torch':
            data_ = self.data
        else:
            data_ = self.data
        return data_

class FlowDiff(Flowfield):
    def __init__(self):
        super.__init__()
    
    def 