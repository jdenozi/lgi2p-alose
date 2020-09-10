#!/bin/env python3
# -*- coding: iso8859-1 -*-


from os import *
import shutil
import os
import time
from sys import *
import sys
sys.path.append(os.path.abspath("preprocessing/"))
from audio import audio as audio
from progressbar import *


class preprocessing:
    def __init__(self,audio_files_path:str): 

        if os.path.exists(audio_files_path)== True:
            self.audio_files_path = audio_files_path
            sys.stdout.write('----------Preprocessing begin------------- \n \n')

        else:
            sys.stdout.write("The given pass is wrong or doesn't exist \n")
            sys.exit()

    def getAudioFilesPath(self):
        return self.audio_files_path

    def getAnnotationFiles(self):
        return self.annotation_files

    def getAudioFiles(self):
        path_directory = self.audio_files_path
        files_path = []
        files_name = []

        for path, subdirs, files in os.walk(path_directory):
            for name in files :
                if name.endswith(".wav"):
                    current_file_path = (path, name)
                    files_path.append(current_file_path)
                    files_name.append(name)
        print("-", len(files_name), "files found in the directory", path_directory,'\n')
        return files_path


    #Permet le decoupage des fichiers audios avec le fichier d'annotation associé
    def trimmingAudio(self, sr = 44100, duration=10):

        audio_files_path = self.getAudioFilesPath() 

        features_dir = getcwd()+"/tmp/bulls_audio/audio_boundaries/"
        s_dir = getcwd()+"/tmp/bulls_audio/audio_out/"
        bull_test = getcwd()+"/tmp/bulls_audio/audio_annotated/"

        ext1 = "_boundaries.txt"
        ext2 = "_annotated.txt"

        files_path = self.getAudioFiles()

        
        widgets=['Preprocessing cut audio files : ',Percentage(),' ',Bar(marker = '0',left = '[',right = ']'), ' ', ETA(), ' ', FileTransferSpeed()]

        pbar = ProgressBar(widgets=widgets, maxval = len(files_path))
        pbar.start()
        progression = 0

        for file_path, file_name in sorted(files_path):
            
            print(file_name)

            feature_file_name = os.path.join(features_dir, os.path.splitext(file_name)[0] + ext1)
            annotated_file_name = os.path.join(bull_test, os.path.splitext(file_name)[0] + ext2)

            audio_tmp = audio(file_path,file_name)

            #ICI duration 
            y, sr, start, stop, tmp_list_time_boundaries = audio_tmp.readAudioFile(sr,duration)
            audio_tmp.saveAudioFile(feature_file_name,tmp_list_time_boundaries)
            samples_folder = audio_tmp.createAudioDirectory(s_dir)

            audio_tmp.cuttingAudio(y, sr, samples_folder, start, stop, tmp_list_time_boundaries)
            audio_tmp.createAnnotatedFile(annotated_file_name, tmp_list_time_boundaries, start, stop)

            progression += 1
            pbar.update(progression)


        pbar.finish()

    def createFolder(self, path,folder_name):
        if os.path.isdir(os.path.join(path,folder_name)) == False:
            os.mkdir(os.path(path,folder_name))










        
        



