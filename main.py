from seg import Segregator
from atfmrdat import FData
from cnn import FModel
from datapreproc import Handler
from optflow import OFMaker
from nlpext import Reporter
import os
import sys

x = str(input("Do you want to train?(y/n)"))
if x=="y":
    s1,s2,s3,s4 = input("Enter shape of images: ").strip().split(" ")
    sh = (s1,s2,s3,s4)
    
    f = FData()
    initdata = f.preprocLoad(shape= (int(sh[0]),int(sh[1]),int(sh[2])),srcdir="RS_data/")
    auxcsv = f.giveAux()
    dp = int(input("Enter no. of datpoints: "))
    s = Segregator(initdata,dp,128)
    segdata = s.datselector(auxcsv)
    s.datManageCheck()

    of = OFMaker(segdata,128)
    ofdata = of.optflowit("mirror")

    dpp = Handler(ofdata,auxcsv,0.2)
    xtr,xts,ytr,yts = dpp.makeDF()
    
    x2 = str(input("Do you want to make a fresh model?(y/n)"))
    if x2=="y":
        cnn = FModel(shape = sh,xtrain = xtr,xtest = xts,ytrain = ytr,ytest = yts)
        cnn.makeModel()
        cnn.evalModel()
    elif x2=="n":
        cnn = FModel(shape = sh,xtrain = xtr,xtest = xts,ytrain = ytr,ytest = yts)
        cnn.model_load("weights-best.hdf5")
        cnn.makeModel()
    else:
        print("Invalid. Quitting.")
elif x=="n":
    srcdir = str(input("Enter source directory of data to be predicted on."))
    s1,s2,s3,s4 = input("Enter shape of images: ").strip().split(" ")
    sh = (s1,s2,s3,s4)
    auxdir = "imtest.csv"
    f = FData(auxdir)
    initdata = f.preprocLoad(shape= (int(sh[0]),int(sh[1]),int(sh[2])),srcdir=srcdir)
    auxcsv = f.giveAux()
    dp = int(input("Enter no. of datpoints: "))
    s = Segregator(initdata,dp,128)
    segdata = s.datselector(auxcsv)
    s.datManageCheck()

    of = OFMaker(segdata,128)
    ofdata = of.optflowit("mirror")

    dpp = Handler(ofdata,auxcsv,0.2)
    dat = dpp.makeTestingDat()
    dpple = dpp.giveLE()
    
    print("Ready to predict")
    cnn = FModel(shape = sh,xtrain = xtr,xtest = xts,ytrain = ytr,ytest = yts)
    cnn.model_load("weights-best.hdf5")
    pred,conf = cnn.predict(dat,dpple)
    
    df = ps.read_csv(auxdir)
    nlp = Reporter(df,pred,dp,conf)
    nlp.makePDF()
