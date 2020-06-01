from keras.layers import Conv3D, MaxPool3D, Flatten, Dense
from keras.layers import Dropout, Input, BatchNormalization
from sklearn.metrics import confusion_matrix, accuracy_score
from plotly.offline import iplot, init_notebook_mode
from keras.losses import categorical_crossentropy
from keras.optimizers import Adadelta
import plotly.graph_objs as go
from matplotlib.pyplot import cm
from keras.models import Model
import numpy as np
import keras
import h5py
from sklearn.metrics import accuracy_score
from keras.callbacks import ModelCheckpoint

class FModel:    

    def __init__(self,shape,xtrain,xtest,ytrain,ytest):
        self.shape = shape
        self.xtrain = xtrain
        self.xtest = xtest
        ytrain = keras.utils.to_categorical(ytrain, 2)
        ytest = keras.utils.to_categorical(ytest, 2)
        self.ytrain = ytrain
        self.ytest = ytest
  
        input_layer = Input(shape)

        conv_layer1 = Conv3D(filters=8, kernel_size=(4, 4, 4), activation='relu')(input_layer)
        conv_layer2 = Conv3D(filters=16, kernel_size=(4, 4, 4), activation='relu')(conv_layer1)

        pooling_layer1 = MaxPool3D(pool_size=(2, 2, 2))(conv_layer2)

        pooling_layer2 = BatchNormalization()(pooling_layer1)
        flatten_layer = Flatten()(pooling_layer2)

        dense_layer1 = Dense(units=5120, activation='relu')(flatten_layer)
        dense_layer1 = Dropout(0.4)(dense_layer1)
        dense_layer2 = Dense(units=2048, activation='relu')(dense_layer1)
        dense_layer2 = Dropout(0.4)(dense_layer2)
        output_layer2 = Dense(units=2, activation='softmax')(dense_layer2)

        self.model = Model(inputs=input_layer, outputs=output_layer2)

    def makeModel(self):
        filepath="weights-best.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]
        self.model.compile(loss=categorical_crossentropy, optimizer=Adadelta(lr=0.02), metrics=['acc'])
        self.model.fit(x=self.xtrain, y=self.ytrain, batch_size=5, epochs=20, validation_split=0.1, callbacks = callbacks_list,verbose=1)

    def evalModel(self,le):
        pred = self.model.predict(self.xtest)
        pred2 = np.argmax(pred, axis=1)
        print(le.inverse_transform(pred2))

    def model_load(shape,file):
        input_layer = Input((shape,64))

        conv_layer1 = Conv3D(filters=8, kernel_size=(4, 4, 4), activation='relu')(input_layer)
        conv_layer2 = Conv3D(filters=16, kernel_size=(4, 4, 4), activation='relu')(conv_layer1)

        pooling_layer1 = MaxPool3D(pool_size=(2, 2, 2))(conv_layer2)

        pooling_layer2 = BatchNormalization()(pooling_layer1)
        flatten_layer = Flatten()(pooling_layer2)

        dense_layer1 = Dense(units=5120, activation='relu')(flatten_layer)
        dense_layer1 = Dropout(0.4)(dense_layer1)
        dense_layer2 = Dense(units=2048, activation='relu')(dense_layer1)
        dense_layer2 = Dropout(0.4)(dense_layer2)
        output_layer2 = Dense(units=2, activation='softmax')(dense_layer2)

        model = Model(inputs=input_layer, outputs=output_layer2)
        model.load_weights(file)
        print("Model loaded")
        print("Model Summary:\n",model.summary())
        self.model = model
        return model

    def predict(self,dat,le):
        self.model = model_load(self.shape)
        preds = self.model.predict(dat)
        conf = le.inverse_transform(preds)
        pred = np.argmax(final, axis=1)
        return pred,conf
