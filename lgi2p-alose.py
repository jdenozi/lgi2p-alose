#!/bin/env python3
# -*- coding: iso8859-1 -*-

import os
import time
from sys import *
import librosa
import matplotlib as plt
import argparse
#os.path.append(getcwd()+"spectrogram/")
#os.path.append(getcwd()+"preprocessing/")

import utils
from spectrogram import spectrogram
from preprocessing import preprocessing
from preprocessing import audio
from spectrogram import melSpectrogram
from models import Model
from models import cnn
from models import vgg16
utils.createFolder()

parser = argparse.ArgumentParser(description='lgi2p-alose')

parser.add_argument('-s','--spectrogram',help='Get all wav audio files from directory and compute the spectrogram of each file. t args allow to process png spectrogram, f args only process spectrogram data')

parser.add_argument('-p','--preprocessing',help='Get a directory with audio files and splits them it into a shorter  and create annotated file. If a number is written after the p arg, it will be the size of the cutted audio')

parser.add_argument('-cnn','--convolutionalnetwork',help='Convolutional neural network')

parser.add_argument('-cnnt','--testcnn',help='Test an audio file with convolutional network')


parser.add_argument('-vgg','--vggconvolutionalnetwork',help='Convolutional network with pre-trained model. Args: "a" change all the layer of vgg16, "l" change the last layer ')

parser.add_argument('-as','--analysis',help='Visualizing sounds')




args = parser.parse_args()

'''
Section dedicated to the convolutional neural network

'''
if args.convolutionalnetwork:
    if args.convolutionalnetwork =="b":
        cnn = cnn.cnn()
        cnn.load()
        cnn.launch()

    if args.convolutionalnetwork=="us":
        cnn = cnn.cnn()
        cnn.loadUnderSampling()
        cnn.launch()

    if args.convolutionalnetwork=="os":
        cnn = cnn.cnn()
        cnn.load(True)
        cnn.launch()

    if args.convolutionalnetwork=="uos":
        cnn = cnn.cnn()
        cnn.load(True)
        cnn.launch()

    if args.convolutionalnetwork=="of":
        pass

    if args.convolutionalnetwork =="cm":
        cnn = cnn.cnn()
        cnn.load()
        cnn.testModel("cnn")

    else:
        print("Wrong arg")

if args.testcnn:
    cnn = cnn.cnn()
    cnn.predictAudio(args.testcnn.strip('"'))



if args.vggconvolutionalnetwork:

    if args.vggconvolutionalnetwork=="b":
        vgg = vgg16.vgg()
        vgg.load()
        vgg.launch()

    if args.vggconvolutionalnetwork=="us":
        vgg = vgg16.vgg()
        vgg.loadUnderSampling()
        vgg.launch()

    if args.vggconvolutionalnetwork=="os":
        vgg = vgg16.vgg()
        vgg.load(True)
        vgg.launch()

    if args.vggconvolutionalnetwork=="uos":
        vgg = vgg16.vgg()
        vgg.loadUnderSampling(True)
        vgg.launch()

    if args.vggconvolutionalnetwork=="llb":
        vgg = vgg16.vgg()
        vgg.load()
        vgg.launchLastLayer()

    if args.vggconvolutionalnetwork=="llus":
        vgg = vgg16.vgg()
        vgg.loadUnderSampling()
        vgg.launchLastLayer()

    if args.vggconvolutionalnetwork=="llos":
        vgg = vgg16.vgg()
        vgg.load(True)
        vgg.launchLastLayer()

    if args.vggconvolutionalnetwork=="lluos":
        vgg = vgg16.vgg()
        vgg.loadUnderSampling(True)
        vgg.launchLastLayer()

    if args.convolutionalnetwork =="cm":
        cnn = cnn.cnn()
        cnn.load()
        cnn.testModel("vgg")








    else :
        print("Wrong arg")

if args.spectrogram:
    utils.deleteMelSpectrogram()
    if args.spectrogram[0]:
        if args.spectrogram[0]=='t':
            spectrogram = spectrogram.spectrogram()
            spectrogram.melSpectrogram(True)
        if args.spectrogram[0]=='f':
            spectrogram = spectrogram.spectrogram()
            spectrogram.melSpectrogram(False)
        else:
            print("Errorr wrong arg")
            exit()
    else:
        spectrogram = spectrogram.spectrogram()
        spectrogram.melSpectrogram()

if args.preprocessing:
    utils.deletePreprocessing()
    preprocessing = preprocessing.preprocessing(args.preprocessing.strip('"'))

    duration = input("Type the length of cutted audio. Or push enter for default duration (10 s): ")

    if len(duration)==0:
        preprocessing.trimmingAudio()
    else:
        duration = int(duration)
        preprocessing.trimmingAudio(44100,duration)

if args.analysis:
    #try:
    x,sr = librosa.load(args.analysis.strip('"'))
    utils.inspect_data(x)
    #except:
    #print("Path error")
