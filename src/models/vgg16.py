#!bin/env python3
# -*- coding: iso8859-1 -*-

import shutil
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import glob
import time
import sys
import utils
from sys import *
from audio import audio as audio
import datetime
from spectrogram import melSpectrogram as melSpectrogram
from progressbar import *
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')
warnings.simplefilter("ignore", category=PendingDeprecationWarning)
import numpy as np
from tensorflow.keras.models import load_model
from memory_profiler import profile
from numba import cuda
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow
import keras
from IPython.display import SVG
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from keras.preprocessing.image import save_img
from keras.utils.vis_utils import model_to_dot
from keras.applications.vgg16 import VGG16,preprocess_input
from keras.applications.xception import Xception
from keras.applications.nasnet import NASNetMobile
from keras.models import Sequential,Input,Model
from tensorflow.keras.layers import Dense,Flatten,Dropout,Concatenate,GlobalAveragePooling2D,Lambda,ZeroPadding2D
from tensorflow.keras.layers import SeparableConv2D,BatchNormalization,MaxPooling2D,Conv2D
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.optimizers import SGD, RMSprop, Adam
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer

from models.Model import Model

class vgg(Model):
    def __init__(self):
        super().__init__()

    def launch(self):

        class_weights = { 0.0 : 1 , 1.0 : 2 }
        inputShape = (128,216,1)
        numClasses = 1

        self.setFeatures(np.array(self.getFeatures()))
        x_train = np.array(self.getFeatures())
        x_test = self.getFeaturesTest()

        print("Train data: ",len(x_train))
        print("Test data : ",len(x_test))

        #split the matrix by each channel in multiples 2D matrices, scale them separately and then put back in 3D format
        scalers = {}
        for i in range(x_train.shape[1]):
            scalers[i] = StandardScaler()
            x_train[:, i, :] = scalers[i].fit_transform(x_train[:, i, :])

        for i in range(x_test.shape[1]):
            x_test[:, i, :] = scalers[i].transform(x_test[:, i, :])


        x_train = x_train[...,np.newaxis]
        y_train = np.asarray(self.getLabels()).astype(np.float32)

        x_test = x_test[...,np.newaxis]
        y_test = np.asarray(self.getLabelsTest()).astype(np.float32)

        data_test = zip(np.array(x_test)[...,np.newaxis],y_test)

        vgg=VGG16(weights=None, include_top=False, input_shape=(128, 431, 1))

        model = tensorflow.keras.Sequential()
        model.add(vgg)
        model.add(GlobalAveragePooling2D())
        model.add(Dense(1024,activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1,activation='sigmoid'))
        model.summary()

        adamOpti = Adam(lr = 0.0001)

        loss = tf.keras.losses.BinaryCrossentropy()
        model.compile(optimizer = adamOpti, loss =loss , metrics = [tf.keras.metrics.BinaryAccuracy(),tf.keras.metrics.Recall()])

        history= model.fit(x=x_train, y=y_train, epochs=20, batch_size=3, validation_data = (x_test, y_test), class_weight=class_weights)
        results = model.evaluate(x_test, y_test, batch_size=5)
        print(results)

    def launchLastLayer(self):
        
        class_weights = { 0.0 : 1 , 1.0 : 4 }
        inputShape = (128,216,1)
        numClasses = 1

        self.setFeatures(np.array(self.getFeatures()))
        x_train = np.array(self.getFeatures())
        x_test = self.getFeaturesTest()

        print("Train data: ",len(x_train))
        print("Test data : ",len(x_test))

        #split the matrix by each channel in multiples 2D matrices, scale them separately and then put back in 3D format
        scalers = {}
        for i in range(x_train.shape[1]):
            scalers[i] = StandardScaler()
            x_train[:, i, :] = scalers[i].fit_transform(x_train[:, i, :])

        for i in range(x_test.shape[1]):
            x_test[:, i, :] = scalers[i].transform(x_test[:, i, :])


        dump(scalers,os.getcwd()+"/tmp/bulls_model/scalers.txt")

        x_train = x_train[...,np.newaxis]
        y_train = np.asarray(self.getLabels()).astype(np.float32)

        x_test = x_test[...,np.newaxis]
        y_test = np.asarray(self.getLabelsTest()).astype(np.float32)

        data_test = zip(np.array(x_test)[...,np.newaxis],y_test)
        
        vgg_conv=VGG16(weights=None, include_top=False, input_shape=(128, 431, 1))

        #Freeze the layers except the last 4 layers
        for layer in vgg_conv.layers[:-4]:
            layer.trainable = False
        
        #Create the model
        model = tensorflow.keras.Sequential()

        #Add the vgg convolutional base model
        model.add(vgg_conv)


        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation="sigmoid"))
            
        adamOpti = Adam(lr = 0.0001)

        loss = tf.keras.losses.BinaryCrossentropy()
        model.compile(optimizer = adamOpti, loss =loss , metrics = [tf.keras.metrics.BinaryAccuracy(),tf.keras.metrics.Recall()])

        history= model.fit(x=x_train, y=y_train, epochs=20, batch_size=3, validation_data = (x_test, y_test), class_weight=class_weights)
        results = model.evaluate(x_test, y_test, batch_size=5)
        print(results)

        model.save(os.getcwd()+"/tmp/bulls_model/model_VGG.h5")


        del model

    def predictAudio(self, path):
    
        audio_tmp = audio(path)
        ext1 = "_boundaries.txt"

        file_name = audio_tmp.getAudioFileName()

        bulls_prediction = os.getcwd()+"/tmp/bulls_prediction/"
        audio_out = bulls_prediction + "audio_out/"

        ''' Delete previous cutted audio files'''
        utils.deleteFiles(audio_out)
        utils.deleteFiles(bulls_prediction + "audio_boundaries/")

        audio_boundaries = os.path.join(bulls_prediction+"/audio_boundaries/", os.path.splitext(file_name)[0] +ext1)

        y, sr, start, stop, tmp_list_time_boundaries = audio_tmp.readAudioFile()

        audio_tmp.saveAudioFile(audio_boundaries,tmp_list_time_boundaries)

        audio_tmp.cuttingAudio(y, sr, audio_out, start, stop, tmp_list_time_boundaries)

        fichier = open(os.getcwd()+"/tmp/bulls_prediction/out.txt", "a")
        for path, subdirs, files in os.walk(audio_out):
            for file in sorted(files) :
                cutted_audio_file_path = os.path.join(audio_out,file)
                cutted_audio_tmp = audio(cutted_audio_file_path)

                mel_spectrogram = melSpectrogram.melSpectrogram(cutted_audio_tmp, audio_out)
                model = tf.keras.models.load_model(os.getcwd()+"/tmp/bulls_model/model_VGG.h5")

                model.compile(loss = 'binary_crossentropy', optimizer = 'rmsprop', metrics = ['accuracy'])

                features = mel_spectrogram.getFeatures()

                if(features.shape!=(128,431)):
                    feature_tmp = features.transpose().copy()
                    feature_tmp.resize((128,431), refcheck=False)
                    features = feature_tmp

                features = features[np.newaxis,:,:,np.newaxis]

                classes = model.predict_classes(features)
                print(file + " - " + str(classes[0][0]))
                fichier.write(file +" - "+str(classes[0][0]))
        fichier.close()
                           
