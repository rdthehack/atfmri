import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

class Handler:
    def __init__(self, data, csv, testsplit, flag = 0):
        self.flag = flag
        self.done = False
        self.data = data
        self.split = testsplit
        self.csv = csv
        
    def makeDF(self):
        sh = self.data.shape
        total = sh[0]
        test_size = int(total*self.split)
        train_size = total-test_size
        x_tr = self.data[:train_size]
        x_ts = self.data[-test_size:]
        y_tr = self.csv["class"][:train_size]
        y_ts = self.csv["class"][-test_size:]
        letr = LabelEncoder()
        ly_tr = letr.fit_transform(y_tr)
        lets = LabelEncoder()
        ly_ts = lets.fit_transform(y_ts)
#         ly_tr = np.argmax(ly_tr, axis=0)
#         ly_ts = np.argmax(ly_ts, axis=0)
        return x_tr,x_ts,ly_tr,ly_ts
        