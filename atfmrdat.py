import os
import numpy as np
import nilearn
from nilearn import image
from nilearn import plotting
import nibabel
from tqdm import tqdm
from skimage.transform import resize
import pandas as pd

class FData:
    
#######################################################################################
    
    def __init__(self,flag=0):
        self.flag = flag
        self.next = ""
        self.prev = ""
        self.auxdata = pd.DataFrame(columns=["name","ino"])
        self.tlist=[]

#######################################################################################     
        
    def getResOne(self, datadir, shape):
        nifti = nibabel.loadsave.load(datadir)
        im = nifti.get_fdata()
        oldsh = im.shape
        res = resize(im[:,:,:,0],shape)
        res = res.astype(np.float16)
        del nifti
        return res,oldsh
    
#######################################################################################     
    
    def makeResData(self, newshape=(28,28,28), srcdir="sampledat/"):
        imshapes = set((0,0,0))
        index = 0
        reached=0
        if srcdir[len(srcdir)-1]!="/":
            srcdir = srcdir+"/"
        else:
            pass
        if srcdir=="sampledat/":
            print("Running on Sample Data")
        else:
            print("Running on Custom Data at Location:",srcdir)
        lt = os.listdir(srcdir)
        for classes in tqdm(lt,desc="Class Iteration:"):
            for _ in os.listdir(srcdir+classes):
                for sub in tqdm(os.listdir(srcdir+classes+"/"+_),desc="Subject Iteration in "+classes+": "):
                    for scan in os.listdir(srcdir+classes+"/"+_+"/"+sub):
                        for date in os.listdir(srcdir+classes+"/"+_+"/"+sub+"/"+scan):
                            for subid in os.listdir(srcdir+classes+"/"+_+"/"+sub+"/"+scan+"/"+date):
                                to_save = np.ndarray(shape = (1,newshape[0],newshape[1],newshape[2]))
                                times = 0
                                for nifti in os.listdir(srcdir+classes+"/"+_+"/"+sub+"/"+scan+"/"+date+"/"+subid):
                                    path = srcdir+classes+"/"+_+"/"+sub+"/"+scan+"/"+date+"/"+subid+"/"+nifti
                                    if nifti[len(nifti)-1]=="i":
                                        resdata,oldshape = self.getResOne(datadir = path, shape = newshape)
                                        resdata = np.expand_dims(resdata,axis=0)
                                        to_save = np.concatenate((to_save,resdata),axis=0)
                                        imshapes.add(tuple(oldshape))
                                        index+=1
                                        reached+=1
                                        times+=1
                                    else:
                                        pass
                                self.tlist.append(times)
                                np.savez_compressed(file = path, img = to_save[:len(to_save)])
        print("Converted Shape: ",newshape)
        print("No. of elements parsed and converted:",reached)
        print("No. of different shapes in data:",len(list(imshapes))-1)
        print("Old shapes encountered and changed:\n",list(imshapes)[1:])
        print(self.tlist)
        
#######################################################################################        

    def loadResData(self, curshape, srcdir, auxdat = 1):
        imgs = np.ndarray(shape = (0,curshape[0],curshape[1],curshape[2]))
        index = 0
        if srcdir[len(srcdir)-1]!="/":
            srcdir = srcdir+"/"
        else:
            pass
        if srcdir=="sampledat/":
            print("Loading Sample Data\n")
        else:
            print("Loading Custom Data from Location:",srcdir)
        lt = os.listdir(srcdir)
        for classes in tqdm(lt,desc="Class Iteration:"):
            for _ in os.listdir(srcdir+classes):
                for sub in tqdm(os.listdir(srcdir+classes+"/"+_),desc="Subject Iteration in "+classes+": "):
                    for scan in os.listdir(srcdir+classes+"/"+_+"/"+sub):
                        for date in os.listdir(srcdir+classes+"/"+_+"/"+sub+"/"+scan):
                            for subid in os.listdir(srcdir+classes+"/"+_+"/"+sub+"/"+scan+"/"+date):
                                for nifti in os.listdir(srcdir+classes+"/"+_+"/"+sub+"/"+scan+"/"+date+"/"+subid):
                                    path = srcdir+classes+"/"+_+"/"+sub+"/"+scan+"/"+date+"/"+subid+"/"+nifti
                                    if nifti[len(nifti)-1]=="z":
                                        img = np.load(file = path)
                                        img = img["img"]
                                        img = img
                                        length = img.shape[0]
                                        imgs = np.concatenate((imgs,img),axis=0)
                                        self.auxdata = self.auxdata.append({"name":nifti,"ino":index,"class":classes,"subject":sub,"type":scan,"date":date,"scanid":subid,"scanshape":img.shape,"len":length},ignore_index=True)
                                        index+=1
                                        del img
                                    else:
                                        pass
        print("Data taken in with Shape: "+str(imgs[1:].shape))
        if auxdat==1:
            self.auxdata.to_csv("image_to_name.csv")
        else:
            pass
        return imgs
        
#######################################################################################     
    
    def preprocLoad(self,shape=(28,28,28), srcdir="sampledat/"):
        self.shape = shape
        self.makeResData(self.shape, srcdir)
        dat = self.loadResData(self.shape, srcdir)
        for i in range(0,dat.shape[0],1):
            dat[i] = (dat[i] - dat[i].min())/(dat[i].max() - dat[i].min())
            dat[i] = dat[i]-0.5
        return dat[1:]
   
#######################################################################################

    def giveAux(self):
        aux = pd.read_csv("image_to_name.csv")
        return aux