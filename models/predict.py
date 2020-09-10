#!bin/env python3
# -*- coding: iso8859-1 -*-
import numpy as np

import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import sys
import os

sys.path.append('../') 
import utils
from preprocessing.audio import audio as audio
from spectrogram import melSpectrogram as melSpectrogram
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model


def predictImage(path):
    path = path[1]
    audio_tmp = audio(path)
    ext1 = "_boundaries.txt"
    ext2 = ".txt"

    file_name = audio_tmp.getAudioFileName()
    
    current_dir,last_dir = os.path.split(os.getcwd())
    bulls_prediction = current_dir+"/tmp/bulls_prediction/"
    audio_out = bulls_prediction + "/audio_out/"
    bulls_annotated = current_dir+"/tmp/bulls_prediction/audio_annotated/"

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

    labels = []
    for path, subdirs, files in os.walk(audio_out):
        for file in sorted(files) :
            cutted_audio_file_path = os.path.join(audio_out,file)
            cutted_audio_tmp = audio(cutted_audio_file_path)

            mel_spectrogram = melSpectrogram.melSpectrogram(cutted_audio_tmp, audio_out)
            mel_spectrogram.computeGraphic("/media/Dossiers/denozi/Documents/Stage_Alose/image_directory/pred/")
            labels.append(dic_Labels.get(file))




    # Load model
    current_dir,last_dir = os.path.split(os.getcwd())

    model = load_model(current_dir+"/tmp/bulls_model/dataGen_CNN.h5")

    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
            "/media/Dossiers/denozi/Documents/Stage_Alose/image_directory/pred/",
            target_size=(128, 216),
            batch_size=20,
            class_mode='binary',
            shuffle=False)

    pred=model.predict_generator(test_generator, steps=len(test_generator), verbose=1)
    results = np.round(pred)


    print(labels)
    print(results)

predictImage(sys.argv)
