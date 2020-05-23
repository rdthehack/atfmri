import os
import numpy as np
import nilearn
from nilearn import image
from nilearn import plotting
import nibabel
from tqdm import tqdm
from skimage.transform import resize
import pandas as pd

class OFMaker:
    
##########################################################################
    
    def __init__(self, data, datpoints, name="opt.csv", axis=1):
        if datpoints%2!=0:
            print("Even datpoints only.")
            return None
        self.data = data
        self.shape = self.data.shape
        self.axis = axis
#         self.frac = frac
        self.name = name
        self.datpoints = datpoints
        
##########################################################################
        
    def decomplexify(self):
        self.dedat = np.ndarray(shape=(0,self.shape[2],self.shape[3],self.shape[4]))
        for _ in range(0,self.shape[0]):
            self.dedat = np.concatenate((self.dedat,self.data[_]))
        print("Decomplexified. Dedat file made.")
     
##########################################################################
    
    def make(self, option = "mirror"):
        if option=="mirror":
            no = self.shape[0]
            offin = np.ndarray(shape = (0,self.shape[2],self.shape[3],self.shape[4],int(self.datpoints/2)))
            for i in range(0,no):
                ofed = np.ndarray(shape = (self.shape[2],self.shape[3],self.shape[4],0))
                tmp = self.dedat[i*self.datpoints:(i+1)*self.datpoints]
                for _ in range(0,int(self.datpoints/2)):
                    cur = int(self.datpoints/2)-1-_
                    tmp2 = tmp[cur] - tmp[_]
                    tmp2 = np.expand_dims(tmp2,axis=3)
                    ofed = np.concatenate((ofed,tmp2),axis=3)
                ofed = np.expand_dims(ofed,axis=0)
                offin = np.concatenate((offin,ofed),axis=0)
            return offin
        elif option=="seq":
            no = self.shape[0]
            offin = np.ndarray(shape = (0,self.shape[2],self.shape[3],self.shape[4],int(self.datpoints/2)))
            for i in range(0,no):
                ofed = np.ndarray(shape = (self.shape[2],self.shape[3],self.shape[4],0))
                tmp = self.dedat[i*self.datpoints:(i+1)*self.datpoints]
                for _ in range(0,int(self.datpoints/2)):
                    tmp2 = tmp[(_*2)+1] - tmp[_*2]
                    tmp2 = np.expand_dims(tmp2,axis=3)
                    ofed = np.concatenate((ofed,tmp2),axis=3)
                ofed = np.expand_dims(ofed,axis=0)
                offin = np.concatenate((offin,ofed),axis=0)
            return offin
        
##########################################################################
    
    def optflowit(self,option):
        self.decomplexify()
        res = self.make(option)
        np.savez_compressed(file="optflowdat.ofdat",ofdat = res)
        return res