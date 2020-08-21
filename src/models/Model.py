#!bin/env python5
# -*- coding: iso8859-1 -*-

import shutil
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import glob
import time
import sys
import utils
from sys import *
import librosa
from audio import audio as audio
import pathlib
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
import json
import random

from keras import backend as K
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv1D, Conv2D, MaxPooling2D, MaxPooling1D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import SGD
from sklearn.metrics import accuracy_score
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix



import numpy as np

class Model:
    def __init__(self):
        self.pathOfFeatures = os.getcwd()+"/tmp/bulls_spectrogram/mel_spectrogram/txt/"
        self.pathOfLabels = os.getcwd()+"/tmp/bulls_audio/audio_annotated/"
        self.path_audio_out = os.getcwd()+"/tmp/bulls_audio/audio_out/"
        self.features = None
        self.labels = None
        self.features_test_path = None
        self.features_test = []
        self.labels_test = []
        self.scaler = None
        self.nb_positive = 0 

    def getPathOfFeatures(self):
        return self.pathOfFeatures

    def getPathOfLabels(self):
        return self.pathOfLabels

    def getFeatures(self):
        return self.features

    def getFeaturesTestPath(self):
        return self.features_test_path


    def getLabels(self):
        return self.labels

    def setFeatures(self,features):
        self.features = features

    def setLabels(self, new_labels):
        self.labels = new_labels

    def getScaler(self):
        return self.scaler

    def setScaler(self,new_scaler):
        self.scaler = new_scaler

    def initFeaturesTest(self):
        path = os.getcwd()+"/config.txt"
        data = None
        with open(path,"r") as f:
            data = json.load(f)
        self.features_test_path = data.get("testFile")

    def getLabelsTest(self): 
        return self.labels_test

    def setLabelsTest(self,new_labels_test):
        self.labels_test = new_labels_test

    def getFeaturesTest(self):
        return self.features_test

    def setFeaturesTest(self, new_features_test):
        self.features_test = new_features_test

    def getNbPositive(self):
        return self.nb_positive

    def setNbPositive(self,new_nb):
        self.nb_positive = new_nb

    def checkPathFeaturesTest(self, path):
        ext = 4
        size_path = len(self.getFeaturesTestPath())-ext
        for i in range(size_path):
            if path[i]!=self.getFeaturesTestPath()[i]:
                return False
        return True

    def positiveAugmentation(self, number_of_positive):
        x = len(self.getFeatures())
        p = number_of_positive/2
        n = x - p
        a = p/n
        for i in range(len(self.getFeatures())):
            if self.getLabels()[i] == "1":
                for i in range(int(a)):

                    feature_tmp = self.getFeatures()[i]
                    feature_tmp = np.random.normal(2*feature_tmp+2)
                    self.getFeatures().append(feature_tmp)
                    self.getLabels().append("1")
        labels = self.getLabels()
        self.setLabels(np.array(labels).astype(np.float))
        del labels

    def testModel(self,type_model):
        if type_model=="cnn":
            model = tf.keras.models.load_model(os.getcwd()+"/tmp/bulls_model/model_CNN.h5")

        if type_model == "vgg":
            model = tf.keras.models.load_model(os.getcwd()+"/tmp/bulls_model/model_VGG.h5")


        x = np.array(self.getFeatures())[...,np.newaxis]
        results = []

        widgets=['Confusion matrix still computing... : ',Percentage(),' ',Bar(marker = '0',left = '[',right = ']'), ' ', ETA(), ' ', FileTransferSpeed()]
        pbar = ProgressBar(widgets=widgets, maxval = len(x))
        pbar.start()
        progression = 0

        for i in x :
            progression += 1
            pbar.update(progression)
            results.append(model.predict_classes( np.array( [i,] )  )[0])
        pbar.finish()
        del model
        y=np.array(self.getLabels()).astype(float)

        results = np.hstack(results).astype(float)

        from sklearn.metrics import classification_report
        from sklearn.metrics import confusion_matrix

        cm=confusion_matrix(y.astype(str),results.astype(str))
        print(cm)

        tn,fp,fn,tp=confusion_matrix(y.astype(str),results.astype(str), normalize="true").ravel()

        target_names=["No Bulls","Bulls"]
        print(classification_report(y.astype(str),results.astype(str), target_names=target_names))
        print("True positive"+str(tp))
        print("True negative"+str(tn))
        print("False positive"+str(fp))
        print("False negative"+str(fn))


    def load(self, aug = False):
        '''
        Read the all spectrogram stored previously and prepare them for the model 

        '''
        self.initFeaturesTest()
        filenames = []
        dic_Labels = {}
        for path, subdirs, files in os.walk(self.getPathOfLabels()):
            for filename in sorted(files) :
                with open(os.path.join(self.getPathOfLabels(),filename),'r') as labels_file:
                    lines = labels_file.readlines()
                    labels_file.close()
                    for line in lines :
                        split_line = line.split(' ')
                        filenames.append(line)
                        file = split_line[0]
                        file_name, extension = os.path.splitext(file)
                        dic_Labels[file_name] = split_line[1][0]

        widgets=['Load audio features and labels : ',Percentage(),' ',Bar(marker = '0',left = '[',right = ']'), ' ', ETA(), ' ', FileTransferSpeed()]
        pbar = ProgressBar(widgets=widgets, maxval = len(dic_Labels))
        pbar.start()
        progression = 0
        filenameD = []
        labels = []
        loop = 0
        features = []
        nb_positive = 0

        for path, subdirs, files in os.walk(self.getPathOfFeatures()):
            for file in sorted(files):
                try:
                    progression += 1
                    pbar.update(progression)
                    current_file = os.path.splitext(file)[0]
                    path_file = os.path.join(path,current_file+".npy")
                    feature_tmp = np.load(path_file)

                    if dic_Labels.get(current_file.replace("_mel_spectrogram",""))!=None:
                        if dic_Labels.get(current_file.replace("_mel_spectrogram",""))=="1":
                            nb_positive += 1 
                        #Part where we stored the spectrogram for the data test
                        if self.checkPathFeaturesTest(file)==True:
                            self.getLabelsTest().append(dic_Labels.get(current_file.replace("_mel_spectrogram","")))
                            #Part where we reshape the audio array if they're not during 10 s
                            if feature_tmp.shape != (128,216):
                                tmp = feature_tmp.transpose().copy()
                                tmp.resize((128,216), refcheck=False)
                                self.getFeaturesTest().append(tmp)
                            else:
                                print('ok')
                                self.getFeaturesTest().append(feature_tmp)
                        #Part where we stored the spectrogram for the training set
                        else:

                            if feature_tmp.shape != (128,216):
                                tmp = feature_tmp.transpose().copy()
                                tmp.resize((128,216), refcheck=False)
                                features.append(tmp)
                                labels.append(dic_Labels.get(current_file.replace("_mel_spectrogram","")))


                            else:
                                features.append(feature_tmp)
                                labels.append(dic_Labels.get(current_file.replace("_mel_spectrogram","")))
                        loop+=1
                except:
                    pass
        pbar.finish()
        self.setFeatures(features)
        self.setLabels(labels)

        self.setFeaturesTest(np.array(self.getFeaturesTest()))
        del features
        del labels
        self.setNbPositive(nb_positive) 
        if aug == True:
            positiveAugmentation(nb_positive)

        
    def loadUnderSampling(self, aug = False):
        self.initFeaturesTest()
        filenames = []
        dic_Labels = {}
        for path, subdirs, files in os.walk(self.getPathOfLabels()):
            for filename in sorted(files) :
                with open(os.path.join(self.getPathOfLabels(),filename),'r') as labels_file:
                    lines = labels_file.readlines()
                    labels_file.close()
                    for line in lines :
                        split_line = line.split(' ')
                        filenames.append(line)
                        file = split_line[0]
                        file_name, extension = os.path.splitext(file)
                        dic_Labels[file_name] = split_line[1][0]
        widgets=['Load audio features and labels : ',Percentage(),' ',Bar(marker = '0',left = '[',right = ']'), ' ', ETA(), ' ', FileTransferSpeed()]
        pbar = ProgressBar(widgets=widgets, maxval = len(dic_Labels))
        pbar.start()
        progression = 0
        filenameD = []
        labels = []
        loop = 0
        features = []
        negative_features = []
        # Store the number of positive feature
        positive_number = 0

        for path, subdirs, files in os.walk(self.getPathOfFeatures()):
            for file in sorted(files):
                    try:
                        progression += 1
                        pbar.update(progression)
                        current_file = os.path.splitext(file)[0]
                        path_file = os.path.join(path,current_file+".npy")
                        feature_tmp = np.load(path_file)

                        if dic_Labels.get(current_file.replace("_mel_spectrogram",""))!=None:

                            if self.checkPathFeaturesTest(current_file)==True:
                                self.getLabelsTest().append(dic_Labels.get(current_file.replace("_mel_spectrogram","")))

                                #Part where we reshape the audio array if they're not during 10 s
                                if feature_tmp.shape != (128,216):
                                    tmp = feature_tmp.transpose().copy()
                                    tmp.resize((128,216), refcheck=False)
                                    self.getFeaturesTest().append(tmp)
                                else:
                                    self.getFeaturesTest().append(feature_tmp)
                                loop+=1

                            else:

                                if dic_Labels.get(current_file.replace("_mel_spectrogram",""))=="1":
                                    labels.append(dic_Labels.get(current_file.replace("_mel_spectrogram","")))
                                    positive_number += 1

                                    #Part where we reshape the audio array if they're not during 10 s
                                    if feature_tmp.shape != (128,216):
                                        tmp = feature_tmp.transpose().copy()
                                        tmp.resize((128,216), refcheck=False)
                                        features.append(tmp)
                                    else:
                                        features.append(feature_tmp)

                                if dic_Labels.get(current_file.replace("_mel_spectrogram",""))=="0":
                                    if feature_tmp.shape != (128,216):
                                        tmp = feature_tmp.transpose().copy()
                                        tmp.resize((128,216), refcheck=False)
                                        negative_features.append([tmp,0])
                                    else:
                                        negative_features.append([feature_tmp,0])
                            loop+=1
                    except:
                        pass
        negatives_features = random.shuffle(negative_features)
        size = len(features)*2

        for i in range(size):
            features.append(negative_features[i][0])
            labels.append(negative_features[i][1])
        pbar.finish()

        self.setFeatures(features)
        print(len(features))
        self.setLabels(labels)

        self.setFeaturesTest(np.array(self.getFeaturesTest()))
        self.setNbPositive(positive_number)
        del features
        del negatives_features
        del labels

        if aug == True:
            self.positiveAugmentation(positive_number)


