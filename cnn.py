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

class FModel:    

    def __init__(self,shape,xtrain,xtest,ytrain,ytest):
        self.shape = shape
        self.xtrain = xtrain
        self.xtest = xtest
        ytrain = keras.utils.to_categorical(ytrain, 2)
        ytest = keras.utils.to_categorical(ytest, 2)
        self.ytrain = ytrain
        self.ytest = ytest
        
        ## input layer
        input_layer = Input(shape)

        ## convolutional layers
        conv_layer1 = Conv3D(filters=8, kernel_size=(4, 4, 4), activation='relu')(input_layer)
        conv_layer2 = Conv3D(filters=16, kernel_size=(4, 4, 4), activation='relu')(conv_layer1)

        ## add max pooling to obtain the most imformatic features
        pooling_layer1 = MaxPool3D(pool_size=(2, 2, 2))(conv_layer2)

        conv_layer3 = Conv3D(filters=32, kernel_size=(4, 4, 4), activation='relu')(pooling_layer1)
        conv_layer4 = Conv3D(filters=64, kernel_size=(4, 4, 4), activation='relu')(conv_layer3)
        pooling_layer2 = MaxPool3D(pool_size=(2, 2, 2))(conv_layer4)

        ## perform batch normalization on the convolution outputs before feeding it to MLP architecture
        pooling_layer2 = BatchNormalization()(pooling_layer2)
        flatten_layer = Flatten()(pooling_layer2)

        ## create an MLP architecture with dense layers : 5120 -> 2048 -> 2
        ## add dropouts to avoid overfitting / perform regularization
        dense_layer1 = Dense(units=5120, activation='relu')(flatten_layer)
        dense_layer1 = Dropout(0.4)(dense_layer1)
        dense_layer2 = Dense(units=2048, activation='relu')(dense_layer1)
        dense_layer2 = Dropout(0.4)(dense_layer2)
        output_layer2 = Dense(units=2, activation='softmax')(dense_layer2)
        ## define the model with input layer and output layer
        self.model = Model(inputs=input_layer, outputs=output_layer2)

    def makeModel(self):
        self.model.compile(loss=categorical_crossentropy, optimizer=Adadelta(lr=0.02), metrics=['acc'])
        self.model.fit(x=self.xtrain, y=self.ytrain, batch_size=1, epochs=20, validation_split=0.1)

    def evalModel(self):
        pred = self.model.predict(self.xtest)
        pred2 = np.argmax(pred, axis=1)
        print(pred)
        print(pred2)
#         print("Accuracy on test set:")
#         print(accuracy_score(self.ytest,pred))
              