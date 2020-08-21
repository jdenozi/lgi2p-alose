#!/bin/env python3
import os
import sys
import librosa
import numpy as np
import matplotlib.pyplot as plt
from spectrogram import spectrogram
from preprocessing import audio

class melSpectrogram():
    def __init__(self, audio, path_out):
        self.audio = audio
        self.path_out = path_out
        self.path_in  = audio.getAudioFilePath() 
        self.y, self.sr = librosa.load(audio.getAudioFullFilePath())
        self.features = librosa.feature.melspectrogram(y=self.getY(), sr=self.getSr())

    def getAudio(self):
        return self.audio

    def getFeatures(self):
        return self.features

    def getY(self):
        return self.y

    def getSr(self):
        return self.sr

    def getPathIn(self):
        return self.path_in

    def getPathOut(self):
        return self.path_out

    def compute(self):
        #self.features.tofile(self.getPathOut())
        np.save(os.path.splitext(self.getPathOut())[0], self.getFeatures())

    def computeGraphic(self, path_out):

        plt.figure(figsize=(5, 5))
        S_dB = librosa.power_to_db(self.getFeatures(), ref=np.max)
        librosa.display.specshow(S_dB, sr=self.getSr(),fmax=8000)
        #plt.colorbar(format='%+2.0f dB')
        #plt.title('Mel-frequency spectrogram')
        plt.tight_layout()
        plt.savefig(path_out)
        plt.close()


