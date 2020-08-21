#!/bin/env python3
# -*- coding: iso8859-1 -*-
import IPython
import matplotlib.pyplot as plt
from matplotlib.pyplot import specgram
import librosa
import shutil
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import glob
import time
import utils
from sys import *
import datetime
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
import tensorflow

import keras
from IPython.display import SVG
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from keras.preprocessing.image import save_img
from keras.utils.vis_utils import model_to_dot
from keras.applications.vgg16 import VGG16,preprocess_input
from keras.applications.xception import Xception
#from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.applications.nasnet import NASNetMobile
from keras.models import Sequential,Input,Model
from tensorflow.keras.layers import Dense,Flatten,Dropout,Concatenate,GlobalAveragePooling2D,Lambda,ZeroPadding2D
from tensorflow.keras.layers import SeparableConv2D,BatchNormalization,MaxPooling2D,Conv2D
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.optimizers import SGD, RMSprop, Adam


import librosa.display

def deleteFiles(pathToDirectory):

    dir = glob.glob(pathToDirectory + "*")
    for file in dir:
        os.remove(file)

def deleteMelSpectrogram():
    mel_spectrogram_dir = os.getcwd()+"/tmp/bulls_spectrogram/"
    for path,subdirs, files in os.walk(mel_spectrogram_dir):
        for name in files:
            os.remove(os.path.join(path,name))

def deletePreprocessing():
    preprocessing = os.getcwd()+"/tmp/bulls_audio/"
    annotation = preprocessing + "audio_annotated/"
    boundaries = preprocessing + "audio_boundaries/"
    out = preprocessing + "audio_out/"
    for path,subdirs, files in os.walk(annotation):
        for name in files:
            print(name)
            os.remove(os.path.join(path,name))

    for path,subdirs, files in os.walk(boundaries):
        for name in files:
            os.remove(os.path.join(path,name))

    for path,subdirs, files in os.walk(out):
        for name in files:
            os.remove(os.path.join(path,name))
    print("Latest preprocessing file deleted")

def createFolder():
    ''' allows the creation of the essential folder'''
    if not os.path.exists(os.getcwd()+"/tmp/"):
        os.makedirs(os.getcwd()+"/tmp/")
    if not os.path.exists(os.getcwd()+"/tmp/bulls_model/"):
        os.makedirs(os.getcwd()+"/tmp/bulls_model/")



    if not os.path.exists(os.getcwd()+"/tmp/bulls_audio/"):
        os.makedirs(os.getwcd()+"/tmp/bulls_audio/")
    if not os.path.exists(os.getcwd()+"/tmp/bulls_audio/audio_annotated/"):
        os.makedirs(os.getcwd()+"/tmp/bulls_audio/audio_annotated/")
    if not os.path.exists(os.getcwd()+"/tmp/bulls_audio/audio_boundaries/"):
        os.makedirs(os.getcwd()+"/tmp/bulls_audio/audio_boundaries/")
    if not os.path.exists(os.getcwd()+"/tmp/bulls_audio/audio_out/"):
        os.makedirs(os.getcwd()+"/tmp/bulls_audio/audio_out/")



    if not os.path.exists(os.getcwd()+"/tmp/bulls_spectrogram/"):
        os.makedirs(os.getwcwd()+"/tmp/bulls_spectrogram/")

    if not os.path.exists(os.getcwd()+"/tmp/bulls_spectrogram/mel_spectrogram/"):
        os.makedirs(os.getcwd()+"/tmp/bulls_spectrogram/mel_spectrogram/")
    if not os.path.exists(os.getcwd()+"/tmp/bulls_spectrogram/mel_spectrogram/txt/"):
        os.makedirs(os.getcwd()+"/tmp/bulls_spectrogram/mel_spectrogram/txt/")

    if not os.path.exists(os.getcwd()+"/tmp/bulls_spectrogram/mel_spectrogram/graph/"):
        os.makedirs(os.getcwd()+"/tmp/bulls_spectrogram/mel_spectrogram/graph/")



    # Creation of the directory for audios prediction
    if not os.path.exists(os.getcwd()+"/tmp/bulls_prediction/"):
        os.makedirs(os.getcwd()+"/tmp/bulls_prediction")

    if not os.path.exists(os.getcwd()+"/tmp/bulls_prediction/audio_boundaries/"):
        os.makedirs(os.getcwd()+"/tmp/bulls_prediction/audio_boundaries/")

    if not os.path.exists(os.getcwd()+"/tmp/bulls_prediction/audio_out/"):
        os.makedirs(os.getcwd()+"/tmp/bulls_prediction/audio_out/")

    if not os.path.exists(os.getcwd()+"/tmp/bulls_prediction/audio_prediction/"):
        os.makedirs(os.getcwd()+"/tmp/bulls_prediction/audio_prediction/")

    if not os.path.exists(os.getcwd()+"/tmp/bulls_prediction/mel_spectrogram/"):
        os.makedirs(os.getcwd()+"/tmp/bulls_prediction/mel_spectrogram/")

#Several function for visualizing sounds
def get_short_time_fourier_transform(soundwave):
    return librosa.stft(soundwave, n_fft=256)

def short_time_fourier_transform_amplitude_to_db(stft):
    return librosa.amplitude_to_db(stft)

def soundwave_to_np_spectogram(soundwave):
    step1 = get_short_time_fourier_transform(soundwave)
    step2 = short_time_fourier_transform_amplitude_to_db(step1)
    step3 = step2/100
    return step3

def inspect_data(sound):
    plt.figure()
    plt.plot(sound)
    #IPython.display.display(IPython.display.Audio(sound, rate=22050))
    a = get_short_time_fourier_transform(sound)
    Xdb = short_time_fourier_transform_amplitude_to_db(a)
    plt.figure()
    plt.imshow(Xdb)
    plt.show()
    print("Length per sample: %d, shape of spectogram: %s, max: %f min: %f" % (len(sound), str(Xdb.shape), Xdb.max(), Xdb.min()))





