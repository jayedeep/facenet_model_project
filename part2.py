#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 13:39:34 2020

@author: jay
"""

import numpy as np
import pickle

import mglearn
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC


def Classifier_model():
        
    # load the face dataset
    data = np.load('5-celebrity-faces-embeddings.npz')
    trainX, trainy = data['arr_0'], data['arr_1']
    print('Loaded: ', trainX.shape, trainy.shape)
    
    print("Dataset: train=%d" % (trainX.shape[0]))
    # normalize input vectors
    in_encoder = Normalizer(norm='l2')
    emdTrainX_norm = in_encoder.transform(trainX)
    # label encode targets
    out_encoder = LabelEncoder()
    out_encoder.fit(trainy)
    trainy_enc = out_encoder.transform(trainy)
    
    model = SVC(kernel='linear', probability=True,C=1,random_state=0)
    model.fit(emdTrainX_norm, trainy_enc)
    # predict
    yhat_train = model.predict(emdTrainX_norm)
    
    pkl_filename1 = "pickle_outcoder.pkl"
    with open(pkl_filename1, 'wb') as file:
        pickle.dump(out_encoder, file)
    
    
    pkl_filename = "pickle_model.pkl"
    with open(pkl_filename, 'wb') as file:
        pickle.dump(model, file)
    
    
    # score
    score_train = accuracy_score(trainy_enc, yhat_train)
    # summarize
    print('Accuracy: train=%.3f' % (score_train*100))



# from sklearn.decomposition import PCA
# pca = PCA(n_components= 62) #We will set it none so that we can see the variance explained and then choose no of comp.
# X_train = pca.fit_transform(emdTrainX_norm)
# X_test = pca.transform(emdTrainX_norm)

# explained_variance = pca.explained_variance_ratio_
# explained_variance

# fit model


#Create classifier object
#classifier_svm_kernel = SVC(C=1.0,kernel='rbf', gamma=0.1,tol=0.00001)
#classifier_svm_kernel.fit(X_train,trainy_enc)

# Grid search and k fold validation libraries already imported. So start the grid search
# from sklearn.model_selection import GridSearchCV
# parameters=[{'C':[1,10,100,1000],'kernel':['linear']},
#             {'C':[1,10,100,1000],'kernel':['rbf'],'gamma':[0.5,0.1,0.01,0.001,0.0001]}
#             ]
# grid_search = GridSearchCV(estimator=classifier_svm_kernel, param_grid=parameters, scoring ='accuracy',cv=10,n_jobs=-1)
# grid_search = grid_search.fit(X_train,trainy_enc)
# best_accuracy=grid_search.best_score_
# best_parameters=grid_search.best_params_


#----------------------------------------


# from numpy import load
# from sklearn.metrics import accuracy_score
# from sklearn.preprocessing import LabelEncoder
# from sklearn.preprocessing import Normalizer
# from sklearn.svm import SVC

# # load dataset
# data = load('5-celebrity-faces-embeddings.npz')
# trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
# print('Dataset: train=%d, test=%d' % (trainX.shape[0], testX.shape[0]))
# # normalize input vectors
# in_encoder = Normalizer(norm='l2')
# trainX = in_encoder.transform(trainX)
# testX = in_encoder.transform(testX)
# # label encode targets
# out_encoder = LabelEncoder()
# out_encoder.fit(trainy)
# trainy = out_encoder.transform(trainy)
# testy = out_encoder.transform(testy)
# # fit model
# model = SVC(kernel='linear', probability=True,C=1,random_state=0)
# model.fit(trainX, trainy)
# # predict
# yhat_train = model.predict(trainX)
# yhat_test = model.predict(testX)
# # score
# score_train = accuracy_score(trainy, yhat_train)
# score_test = accuracy_score(testy, yhat_test)
# # summarize
# print('Accuracy: train=%.3f, test=%.3f' % (score_train*100, score_test*100))



#from sklearn.model_selection import cross_val_score
#accuracies=cross_val_score(estimator=model,X=emdTrainX_norm,y=trainy_enc,cv=10)

#from sklearn.model_selection import GridSearchCV
#parameters=[{'C':[1,10,100,1000],'kernel':['linear']},
 #           {'C':[1,10,100,1000],'kernel':['rbf'],'gamma':[0.5,0.1,0.01,0.001,0.0001]}
  #          ]
#grid_search=GridSearchCV(estimator=model,param_grid=parameters,scoring='accuracy',cv=10)
#grid_search=grid_search.fit(emdTrainX_norm,trainy_enc)
#best_accuracy=grid_search.best_score_
#best_parameters=grid_search.best_params_


