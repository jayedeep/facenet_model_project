#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 13:58:25 2020

@author: jay
"""

import datetime

start=datetime.datetime.now()
import numpy as np
from PIL import Image
from mtcnn.mtcnn import MTCNN
from keras.models import load_model
import pickle

import cv2
import numpy as np
from os import listdir 
from os.path import isfile, join
import csv
from makecsv import record_data

user='newuser'


def Main_test():
    name_list= []
   
    filename = "class_records.csv"

    with open(filename, 'a') as csvfile:
        pass   


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
    
    
    facenet_model = load_model('facenet_keras.h5')
    
    with open("pickle_model.pkl", 'rb') as file:
        model = pickle.load(file)
    #----------------------------------------------------------
    model2 =cv2.face.LBPHFaceRecognizer_create()
    model2.read("detection.xml")
    #------------------------------------------------------------
    
    with open("pickle_outcoder.pkl", 'rb') as file:
        out_encoder = pickle.load(file)
    
    #_--------------------------------------------------------------------
    
    face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    
    
    def face_detector(img, size = 0.5):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        faces = face_classifier.detectMultiScale(gray,1.3,5)
        
        if faces is():
            return img,[]
        
        for(x,y,w,h) in faces:
            cv2.rectangle(img, (x,y),(x+w,y+h),(0,255,255),2)
            roi = img[y:y+h, x:x+w]
            roi = cv2.resize(roi, (200,200))
            
        return img,roi
    
    list_attendance=[]
    
    cap = cv2.VideoCapture(0)
    while True:
        
        ret, frame = cap.read()
        
        image, face = face_detector(frame)
        
        try:
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            result = model2.predict(face)
            
            if result[1] < 500:
                confidence = int(100*(1-(result[1])/300))
                display_string = str(confidence)+'%Confidence he is user'
            cv2.putText(image,display_string,(100,120),cv2.FONT_HERSHEY_COMPLEX,1,(250,120,255),2)
            
            if confidence > 75:
    #             cv2.imshow('Face Cropper', image)
                 file_test_path = 'test_pic/'+str(user)+'.jpg';
                 cv2.imwrite(file_test_path,image)
                 # select a random face from test set
                 test_pic = extract_face(file_test_path)
                 example_face_emd = get_embedding(facenet_model, test_pic)
                
                 # prediction for the face
                 face_predict = np.expand_dims(example_face_emd, axis=0)
                 yhat_class = model.predict(face_predict)
                 yhat_prob = model.predict_proba(face_predict)
                
                 # get name
                 class_index = yhat_class[0]
                 class_probability = yhat_prob[0,class_index] * 100
                 predict_names = out_encoder.inverse_transform(yhat_class)
                 all_names = out_encoder.inverse_transform(list(range(len(yhat_prob.reshape((yhat_prob.T.shape))))))
                
                 #print('Predicted: %s (%.3f)' % (predict_names[0], class_probability))
                 
                 
                 if class_probability>=99.99:
                     names=predict_names[0]
                     if names not in name_list:
                         name_list.append(names)
                         
                         record_data([names])
                        
                     
                 else:
                     names='unknows'
                     
                 print('Predicted: \n%s \n%s' % (names, class_probability))
    
#                 list_attendance.append(names)
                 
                 
                 cv2.putText(image,str(names), (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                 cv2.imshow('Face Cropper', image)
                 end=datetime.datetime.now() 
    
    
            else:
                cv2.putText(image, "Incorrect", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow('Face Cropper', image)
          
            
            
        except:
            cv2.putText(image,"Face Not Found",(250, 450),cv2.FONT_HERSHEY_COMPLEX,1,(255, 0, 0), 2)
            cv2.imshow('Face Cropper', image)
            pass
        if cv2.waitKey(1)==13:
            break
        
        
    cap.release()
    cv2.destroyAllWindows()
   # list_attendance=list(set(list_attendance))

    #pkl_filenames = "list_attendance1.pkl"
    #with open(pkl_filenames, 'wb') as file:
     #   pickle.dump(list_attendance, file)
    
Main_test()

    
    #test_pic = extract_face("test_photo/test2.jpg")
    #example_face_emd = get_embedding(facenet_model, test_pic)
                
                 # prediction for the face
    #face_predict = np.expand_dims(example_face_emd, axis=0)
    #yhat_class = model.predict(face_predict)
    #yhat_prob = model.predict_proba(face_predict)
                
    # get name
    #class_index = yhat_class[0]
    #class_probability = yhat_prob[0,class_index] * 100
    #predict_names = out_encoder.inverse_transform(yhat_class)
    #print('Predicted: %s (%.3f)' % (predict_names[0], class_probability))
    
    #end=datetime.datetime.now()
    #all_names = out_encoder.inverse_transform([0,1,2,3])
                
    #print('Predicted: %s (%.3f)' % (predict_names[0], class_probability))
    #print('Predicted: \n%s \n%s' % (all_names, yhat_prob[0]*100))
    
    
