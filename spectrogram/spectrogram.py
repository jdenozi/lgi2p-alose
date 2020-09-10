#!/bin/env python3
import os
from os import * 
import sys
import shutil
import librosa
from progressbar import *
sys.path.append(os.path.abspath("preprocessing/"))
from preprocessing.audio import audio as audio
from spectrogram import melSpectrogram

class spectrogram:
    def __init__(self):
        self.path_audio_out = getcwd()+"/tmp/bulls_audio/audio_out/"
        self.path_spectrogram = getcwd()+"/tmp/bulls_spectrogram/"

    def getPathAudioOut(self):
        return self.path_audio_out

    def getPathSpectrogram(self):
        return self.path_spectrogram

    def melSpectrogram(self, graph = False):
        audio_files_path = self.getAudioFiles()
        widgets=['Compute mel-spectrogram : ',Percentage(),' ',Bar(marker = '0',left = '[',right = ']'), ' ', ETA(), ' ', FileTransferSpeed()]
        pbar = ProgressBar(widgets=widgets, maxval = len(audio_files_path))
        pbar.start()
        progression = 0
        files_path = self.getAudioFiles()

        mel_spectrogram_graph_dir = getcwd()+"/tmp/bulls_spectrogram/mel_spectrogram/graph/"
        mel_spectrogram_txt_dir = getcwd()+"/tmp/bulls_spectrogram/mel_spectrogram/txt/"


        self.createFolderMelSpectrogram() 


        for file_path, file_name in audio_files_path:
            audio_tmp = audio(file_path, file_name)

            self.createFolder(mel_spectrogram_txt_dir, audio_tmp.getDir())

            path_subdir = os.path.join(mel_spectrogram_txt_dir, audio_tmp.getDir(), os.path.splitext(audio_tmp.getAudioFileName())[0]+"_mel_spectrogram.txt")
            
            mel_spectrogram_tmp = melSpectrogram.melSpectrogram(audio_tmp, path_subdir)

            
            #Create subdir if not exist
            mel_spectrogram_tmp.compute()

            if graph == True:
                path_graph_subdir = os.path.join(mel_spectrogram_graph_dir, audio_tmp.getDir(), os.path.splitext(audio_tmp.getAudioFileName())[0]+"_mel_spectrogram.png")
                self.createFolder(mel_spectrogram_graph_dir, audio_tmp.getDir())
                mel_spectrogram_tmp.computeGraphic(path_graph_subdir)
            else:
                pass
            progression += 1
            pbar.update(progression)
        pbar.finish()


    def createFolder(self, path,folder_name):
        if os.path.isdir(os.path.join(path,folder_name)) == False:
            os.mkdir(os.path.join(path,folder_name))


    def getAudioFiles(self):
        path_directory = self.getPathAudioOut()
        files_path = []
        files_name = []

        for path, subdirs, files in os.walk(path_directory):
            for name in files :
                if name.endswith(".wav"):
                    current_file_path = (path, name)
                    files_path.append(current_file_path)
                    files_name.append(name)
        return files_path

    def createFolderMelSpectrogram(self):
        if os.path.isdir(os.path.join(self.getPathSpectrogram(),"mel_spectrogram/")) == False:
                os.mkdir(os.path.join(self.getPathSpectrogram(),"mel_spectrogram/"))
                os.mkdir(os.path.join(self.getPathSpectrogram(),"mel_spectrogram/txt/"))
                os.mkdir(os.path.join(self.getPathSpectrogram(),"mel_spectrogram/graph/"))



