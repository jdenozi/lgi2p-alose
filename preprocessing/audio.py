#!bin/env python3
# -*- coding: iso8859-1 -*-


import os
import sys
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import warnings
import pathlib
import soundfile as sf

if not sys.warnoptions:
    warnings.simplefilter("ignore")

#audio_file_path:str, audio_file_name:str
#audio_full path
class audio:
    def __init__(self, *args):
        if len(args)==2:
            self.audio_file_path = args[0]
            self.audio_file_name = args[1]
            self.audio_full_file_path = os.path.join(self.audio_file_path,self.audio_file_name)
            self.dir = self.collectDirNSubdir(args[0])

        if len(args)==1:
            self.audio_full_file_path = args[0]
            self.audio_file_name = os.path.basename(args[0])
            self.audio_file_path = os.path.splitext(args[0])


    def getAudioFilePath(self):
        return self.audio_file_path

    def getAudioFileName(self):
        return self.audio_file_name

    def getAudioFullFilePath(self):
        return self.audio_full_file_path

    def getDir(self):
        return self.dir
    
    def collectDirNSubdir(self, audio_file_path):
        folders = []

        while 1:
            path, folder = os.path.split(audio_file_path)

            if folder != "":
                folders.append(folder)
                audio_file_path = path
            else:
                if path != "":
                    folders.append(path)
                break
        dir = folders[0]
        return dir

   #Compute the step boundaries 
    def save_boundaries(self,start, stop):
        n = len(start)
        m = 2
        l = [[0]*m for i in range(n)]
        for i in range(n):
            l[i][0] = start[i]
            l[i][1] = stop[i]
        return l

    #Create the boundaries of each cutted audio 
    def readAudioFile(self,sr = 44100, duration=10):
        frame_length = duration * sr
        hop_length = duration * sr

        y,sr = librosa.load(self.getAudioFullFilePath(),sr)
        start = np.arange(0,len(y), hop_length) / sr
        stop = [x + (frame_length/float(sr)) for x in start]

        list_time_boundaries = self.save_boundaries(start,stop)
        try :
            y = y_mono.shape
        except:
            pass

        return y, sr, start, stop, list_time_boundaries

    
    def saveAudioFile(self, directory, list_time_boundaries):
        ''' Save the all time boundaries in a text file'''

        np.savetxt(directory,list_time_boundaries,fmt='%10d', delimiter=',')

    
    #Create the directory where the audio is stored
    def createAudioDirectory(self,directory_audio_cut):
        samples_folder = os.path.join(directory_audio_cut,os.path.splitext(self.getAudioFileName())[0])
        try :
            os.makedirs(samples_folder)
        except :
            pass
        return samples_folder
    
    def cuttingAudio(self,y, sr, samples_folder, start, stop,list_time_boundaries):
        for i in range(len(list_time_boundaries)):
            x = y[int(list_time_boundaries[i][0]) * sr : int(list_time_boundaries[i][1]) * sr]
            filename = os.path.join(samples_folder, os.path.splitext(self.getAudioFileName())[0] + "_" + str(int(start[i])).zfill(4) + '.wav')
            #librosa.output.write_wav(filename, x, sr)
            sf.write(filename, x, sr)
        
    def createAnnotatedFile(self, annotated_file_name, list_time_boundaries, start, stop):
        line_number = len(list_time_boundaries)
        column_number = 2
        annotated_files = [[0] * column_number for i in range(line_number)]

        for i in range(line_number):
            annotated_files[i][0] = os.path.splitext(self.getAudioFileName())[0] + "_" + str(int(start[i])).zfill(4) + '.wav'

        with open('{}.txt'.format(os.path.splitext(self.getAudioFullFilePath())[0])) as current_audio_file:
                annotated_file = np.loadtxt(current_audio_file)

                for i in range(line_number) :
                    for j in range(len(annotated_file)):
                        if(start[i] <= annotated_file[j][0] <= stop[i]) or (start[i] <= annotated_file[j][1] <= stop[i]):
                            annotated_files[i][1] = "1"
                np.savetxt(annotated_file_name, annotated_files, fmt='%s')
                



        
            


        
