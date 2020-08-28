#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 13:35:06 2020

@author: jay
"""



import mtcnn
# print version
print(mtcnn.__version__)

import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from keras.preprocessing import image
from os import listdir
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
from os.path import isfile, join
import pickle


from mtcnn.mtcnn import MTCNN
from keras.models import load_model

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
#print(os.listdir())

from part2 import Classifier_model

def main_train():
    
    # extract a single face from a given photograph
    def extract_face(filename, required_size=(160, 160)):
        # load image from file
        image = Image.open(filename)
        # convert to RGB, if needed
        image = image.convert('RGB')
        # convert to array
        pixels = np.asarray(image)
        # create the detector, using default weights
        detector = MTCNN()
        # detect faces in the image
        results = detector.detect_faces(pixels)
        # extract the bounding box from the first face
        x1, y1, width, height = results[0]['box']
        # deal with negative pixel index
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        # extract the face
        face = pixels[y1:y2, x1:x2]
        # resize pixels to the model size
        image = Image.fromarray(face)
        image = image.resize(required_size)
        face_array = np.asarray(image)
        return face_array
    
    
    def load_face(dir):
        faces = list()
        # enumerate files
        for filename in os.listdir(dir):
            path = dir + filename
            face = extract_face(path)
            faces.append(face)
        return faces
    
    def load_dataset(dir):
        # list for faces and labels
        X, y = list(), list()
        for subdir in os.listdir(dir):
            path = dir + subdir + '/'
            faces = load_face(path)
            labels = [subdir for i in range(len(faces))]
            print("loaded %d sample for class: %s" % (len(faces),subdir) ) # print progress
            X.extend(faces)
            y.extend(labels)
        return np.asarray(X), np.asarray(y)
    
    #load train dataset
    trainX, trainy = load_dataset('5-celebrity-faces-dataset/train/')
    print(trainX.shape, trainy.shape)
    #save and compress the dataset for further use
    np.savez_compressed('5-celebrity-faces-dataset.npz', trainX, trainy)
    
    #load the face dataset
    data = np.load('5-celebrity-faces-dataset.npz')
    trainX, trainy = data['arr_0'], data['arr_1']
    print('Loaded: ', trainX.shape, trainy.shape)
    
    
    # trainX, trainy = load_dataset('5-celebrity-faces-dataset/train/')
    # print(trainX.shape, trainy.shape)
    # # load test dataset
    # testX, testy = load_dataset('5-celebrity-faces-dataset/val/')
    # print(testX.shape, testy.shape)
    # # save arrays to one file in compressed format
    # np.savez_compressed('5-celebrity-faces-dataset.npz', trainX, trainy, testX, testy)
     
    # # load the face dataset
    # data = np.load('5-celebrity-faces-dataset.npz')
    # trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
    # print('Loaded: ', trainX.shape, trainy.shape, testX.shape, testy.shape)
    
    
    
    #load the facenet model
    facenet_model = load_model('facenet_keras.h5')
    print('Loaded Model')
    
    
    def get_embedding(model, face):
        # scale pixel values
        face = face.astype('float32')
        # standardization
        mean, std = face.mean(), face.std()
        face = (face-mean)/std
        # transfer face into one sample (3 dimension to 4 dimension)
        sample = np.expand_dims(face, axis=0)
        # make prediction to get embedding
        yhat = model.predict(sample)
        return yhat[0]
        
    # convert each face in the train set into embedding
    emdTrainX = list()
    for face in trainX:
        emd = get_embedding(facenet_model, face)
        emdTrainX.append(emd)
        
    emdTrainX = np.asarray(emdTrainX)
    print(emdTrainX.shape)
    
    
    # newTestX = list()
    # for face_pixels in testX:
    # 	embedding = get_embedding(facenet_model, face_pixels)
    # 	newTestX.append(embedding)
    # newTestX = asarray(newTestX)
    # print(newTestX.shape)
    
    # np.savez_compressed('5-celebrity-faces-embeddings.npz', newTrainX, trainy, newTestX, testy)
    # 
    np.savez_compressed('5-celebrity-faces-embeddings.npz', emdTrainX, trainy)
    #-----------------------------------------------------------------------------
    face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    
    data_path = 'cm_100_pic/'
    onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]
    
    Training_Data, Labels = [], []
     
    for i, files in enumerate(onlyfiles):
        image_path = data_path + onlyfiles[i]
        images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        Training_Data.append(np.asarray(images, dtype=np.uint8))
        Labels.append(i)
         
    Labels = np.asarray(Labels, dtype=np.int32)
    
    model2 =  cv2.face.LBPHFaceRecognizer_create()
    model2.train(np.asarray(Training_Data), np.asarray(Labels))
    
    print("Model Training Complete!!!!!")
    
    model2.save('detection.xml')

main_train()
Classifier_model()