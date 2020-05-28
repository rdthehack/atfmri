from seg import Segregator
from atfmrdat import FData
from cnn import FModel
from datapreproc import Handler
from optflow import OFMaker
import os
import sys

f = FData()
initdata = f.preprocLoad(shape= (24,24,24),srcdir="RS_data/")
auxcsv = f.giveAux()

s = Segregator(initdata,4,128)
segdata = s.datselector(auxcsv)
s.datManageCheck()

of = OFMaker(segdata,4)
ofdata = of.optflowit("mirror")

dpp = Handler(ofdata,auxcsv,0.2)
xtr,xts,ytr,yts = dpp.makeDF()

cnn = FModel(shape = (36,36,36,2),xtrain = xtr,xtest = xts,ytrain = ytr,ytest = yts)
cnn.makeModel()
cnn.evalModel()
