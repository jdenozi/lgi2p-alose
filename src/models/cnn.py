#!bin/env python5
# -*- coding: iso8859-1 -*-

import shutil
import tensorflow_addons as tfa
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import glob
import random
import time
import sys
import utils
from sys import *
import librosa
from audio import audio as audio
import pathlib
import pandas as pd
import datetime
from spectrogram import melSpectrogram as melSpectrogram
from progressbar import *
from sklearn.preprocessing import normalize 
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')
warnings.simplefilter("ignore", category=PendingDeprecationWarning)
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import tensorflow.keras as keras
from keras.utils import to_categorical
import keras
from keras import backend as K
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense, Dropout, Flatten,Input
from tensorflow.keras.layers import Conv1D, Conv2D, MaxPooling2D, MaxPooling1D
#from tensorflow.keras.layers.normalization import BatchNormalization
#from tensorflow.keras.layers.advanced_activations import LeakyReLU
from tensorflow.keras.optimizers import SGD, Adam, RMSprop, Adadelta, Adagrad, Adamax
from sklearn.metrics import accuracy_score
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
import json
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
import joblib
from joblib import dump, load
from models.Model import Model
import numpy as np

class cnn(Model):
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

            
        dump(scalers,os.getcwd()+"/tmp/bulls_model/scalers.txt")
        
        x_train = x_train[...,np.newaxis]
        y_train = np.asarray(self.getLabels()).astype(np.float32)

        x_test = x_test[...,np.newaxis]
        y_test = np.asarray(self.getLabelsTest()).astype(np.float32)

        data_test = zip(np.array(x_test)[...,np.newaxis],y_test)
       
        model = Sequential()
        model.add(Conv2D(16, kernel_size=(3, 3),activation='relu',input_shape=inputShape,padding='same'))
        model.add(MaxPooling2D())

        model.add(Conv2D(32, (3, 3), activation='relu',padding='same'))
        model.add(MaxPooling2D())

        model.add(Conv2D(64, (3, 3), activation='relu',padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(128, (3, 3), activation='relu',padding='same'))
        model.add(MaxPooling2D())

        model.add(Conv2D(256, (3, 3), activation='relu',padding='same'))
        model.add(MaxPooling2D())
        model.add(Flatten())

        model.add(Dense(512,activation='relu'))
        model.add(Dropout(0.5))

        model.add(Dense(1, activation='sigmoid'))
        adamOpti = Adam(lr = 0.0001)

        loss = tf.keras.losses.BinaryCrossentropy()
        model.compile(optimizer = adamOpti, loss =loss , metrics = [tf.keras.metrics.BinaryAccuracy(),tf.keras.metrics.Recall()])

        history= model.fit(x=x_train, y=y_train, epochs=20, batch_size=3, validation_data = (x_test, y_test), class_weight=class_weights)
        results = model.evaluate(x_test, y_test, batch_size=5)   
        print(results)

        model.save(os.getcwd()+"/tmp/bulls_model/model_CNN.h5")


    def predictAudio(self, path):
        audio_tmp = audio(path)
        ext1 = "_boundaries.txt"
        ext2 = ".txt"

        #self.setScaler(load(os.getcwd()+"/tmp/bulls_model/scalers.txt"))


        file_name = audio_tmp.getAudioFileName()

        bulls_prediction = os.getcwd()+"/tmp/bulls_prediction/"
        audio_out = bulls_prediction + "audio_out/"
        bulls_annotated = os.getcwd()+"/tmp/bulls_prediction/audio_annotated/"

        ''' Delete previous cutted audio files'''
        utils.deleteFiles(audio_out)
        utils.deleteFiles(bulls_prediction + "audio_boundaries/")
 
        audio_boundaries = os.path.join(bulls_prediction+"/audio_boundaries/", os.path.splitext(file_name)[0] +ext1)

        y, sr, start, stop, tmp_list_time_boundaries = audio_tmp.readAudioFile()

        audio_tmp.saveAudioFile(audio_boundaries,tmp_list_time_boundaries)

        audio_tmp.cuttingAudio(y, sr, audio_out, start, stop, tmp_list_time_boundaries)

        annotated_file = np.loadtxt(os.path.splitext(path)[0]+ext2)

        line_number = len(tmp_list_time_boundaries)
        column_number = 2
        annotated_files = [[0] * column_number for i in range(line_number)]

        for i in range(line_number):
            annotated_files[i][0] = os.path.splitext(os.path.basename(path))[0] + "_" + str(int(start[i])).zfill(4) + '.wav'

        for i in range(line_number):
            for j in range(len(annotated_file)):
                if(start[i] <= annotated_file[j][0] <= stop[i]) or (start[i] <= annotated_file[j][1] <= stop[i]):
                    annotated_files[i][1] = "1"

        dic_Labels = {}
        for i in annotated_files:
            dic_Labels[i[0]]=i[1]

        all_features = []
        labels = []

        widgets=['Audio preprocessing still computing : ',Percentage(),' ',Bar(marker = '0',left = '[',right = ']'), ' ', ETA(), ' ', FileTransferSpeed()]
        pbar = ProgressBar(widgets=widgets, maxval = len(dic_Labels))
        pbar.start()
        progression = 0

        for path, subdirs, files in os.walk(audio_out):
            for file in sorted(files) :
                progression += 1
                pbar.update(progression)

                cutted_audio_file_path = os.path.join(audio_out,file)
                cutted_audio_tmp = audio(cutted_audio_file_path)

                mel_spectrogram = melSpectrogram.melSpectrogram(cutted_audio_tmp, audio_out)

                features = mel_spectrogram.getFeatures()
                
                model = tf.keras.models.load_model(os.getcwd()+"/tmp/bulls_model/model_CNN.h5")

                model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
                
                if features.shape != (128,216):
                    tmp = features.transpose().copy()
                    tmp.resize((128,216), refcheck=False)
                    all_features.append(tmp)
                else:
                    all_features.append(features)
                labels.append(dic_Labels.get(file))
        pbar.finish()
        

        '''
        Predicting part

        '''
        x_test = np.array(all_features)
        labels = np.array(labels).astype(np.float)
        p = model.predict_classes(x_test[...,np.newaxis]) 
        p = np.hstack(p).astype(np.float)


        time = 0
        for i in range(len(labels)):
            print(time,labels[i],p[i])
            time+=10

        from sklearn.metrics import classification_report
        from sklearn.metrics import confusion_matrix
        cm=confusion_matrix(labels.astype(str),p.astype(str))


        
        print(cm)
        tn,fp,fn,tp=confusion_matrix(labels.astype(str),p.astype(str),normalize="true").ravel()

        target_names=["No Bulls","Bulls"]

        print("True positive"+str(tp))
        print("True negative"+str(tn))
        print("False positive"+str(fp))
        print("False negative"+str(fn))
        
        target_names=["Bulls","No Bulls"]
        print(classification_report(labels,p, target_names=target_names))








