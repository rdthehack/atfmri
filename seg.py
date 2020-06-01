import os
import numpy as np
import nilearn
from nilearn import image
from nilearn import plotting
import nibabel
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

class Segregator:
    
#############################################################################################
    
    def __init__(self, data, datpoints, lenthresh, flag = 0):
        if datpoints%2!=0:
            print("datpoints are odd. Invalid.")
            return None
        print("Continuing with",datpoints,"Data points.")
        shape = data.shape
        self.thresh = lenthresh
        self.datpoints = datpoints
        self.shape = shape
        self.x = shape[1]
        self.y = shape[2]
        self.z = shape[3]
        self.flag = flag
        self.aclist = []
        print("Loading given data into object.")
        self.data = data
        print("Loaded data. Proceed with manual flow:\ndatselector -> divideByAxis -> saveData\nOR\nRun segregation to automanage.")
        
#############################################################################################
    
    def divideByAxis(self, imgarr, shape): #2D
        x = shape[0]
        y = shape[1]
        z = shape[2]
        self.x = x
        self.y = y
        self.z = z
        print("Segregating all axial slices")
        for num in tqdm(range(imgarr.shape[0])):
            tax = np.ndarray(shape = (1,y,z))
            tmp = imgarr[num,:,:,:]
            for x_const_val in range(0, x, 1):
                to_concat = tmp[x_const_val,:,:]
                to_concat = np.expand_dims(to_concat, axis=0)
                tax = np.concatenate((tax,to_concat),axis=0)
            tax = tax[1:,:,:]
            tax = np.expand_dims(tax,axis=0)
            self.ax = np.concatenate((self.ax,tax))
        self.ax = self.ax[1:,:,:,:]
        
        print("Segregating all saggital slices")
        for num in tqdm(range(imgarr.shape[0])):
            tsa = np.ndarray(shape = (1,x,z))
            tmp = imgarr[num,:,:,:]
            for y_const_val in range(0, y, 1):
                to_concat = tmp[:,y_const_val,:]
                to_concat = np.expand_dims(to_concat, axis=0)
                tsa = np.concatenate((tsa,to_concat),axis=0)
            tsa = tsa[1:,:,:]
            tsa = np.expand_dims(tsa,axis=0)
            self.sa = np.concatenate((self.sa,tsa))
        self.sa = self.sa[1:,:,:,:]
        
        print("Segregating all coronal slices")
        for num in tqdm(range(imgarr.shape[0])):
            ax = np.ndarray(shape = (1,x,y))
            tmp = imgarr[num,:,:,:]
            for z_const_val in range(0, z, 1):
                to_concat = tmp[:,:,z_const_val]
                to_concat = np.expand_dims(to_concat, axis=0)
                tco = np.concatenate((tco,to_concat),axis=0)
            tco = tco[1:,:,:]
            tco = np.expand_dims(tco,axis=0)
            self.co = np.concatenate((self.co,tco))
        self.co = self.co[1:,:,:,:]
        
#############################################################################################
        
    def getData(self, axis): #2D
        if axis==1:
            return self.ax[1:,:,:]
        elif axis==2:
            return self.sa[1:,:,:]
        elif axis==3:
            return self.co[1:,:,:]
        else:
            print("Axis not valid for 4D segregation")
        
#############################################################################################

    def datselector(self,auxcsv):
        lenlist = auxcsv["len"].tolist()
        seged = np.ndarray(shape = (0,self.datpoints,self.x,self.y,self.z))
        base_index = 0
        rem = 0
        for _ in range(0,len(lenlist),1):
            tmp = np.ndarray(shape = (0,self.x,self.y,self.z))
            thislen = lenlist[_]
            if thislen<=self.thresh:
                rem+=1
                pass
            else:
                iterlen = thislen//self.datpoints
                for i in range(0,self.datpoints,1):
                    cur = int(base_index+(i*iterlen))
                    tms = self.data[cur]
                    tms = np.expand_dims(tms,axis=0)
                    tmp = np.concatenate((tmp,tms))
                tmp = np.expand_dims(tmp,axis=0)
                seged = np.concatenate((seged,tmp))
                base_index+=thislen
        if (len(lenlist)-rem)!=seged.shape[0]:
            print("***WARNING***\nSegregated data length and modified reported length is not equal. Please check integrity.")
        else:
            print("Segmented Successfully")
            print("No. of entries falling under thresh value: ",rem)
        self.flag=1
        return seged
        
#############################################################################################
        
    def datManageCheck(self):
        if self.flag==1:
            print("Data has been checked and verified by datselector. Proceeding.\n")
        elif self.flag==0:
            print("Data has not been checked. Verifying with datselector. Please wait...\n")
            self.datselector()
            print("Data checked and verified by datselector automatically. Proceeding.\n")
        else:
            print("Data check flag invalid or has been corrupted. Please check source.")

#############################################################################################
        
    def saveData(self,no): #2D
        np.savez_compressed(file = "segregated/axial/d"+str(no)+".segdat",segdat = self.ax)
        np.savez_compressed(file = "segregated/saggital/d"+str(no)+".segdat",segdat = self.sa)
        np.savez_compressed(file = "segregated/coronal/d"+str(no)+".segdat",segdat = self.co)
        
#############################################################################################

    def segregation(self,shape):
        pass
        
    
#############################################################################################
    
