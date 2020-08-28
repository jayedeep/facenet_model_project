#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 11:48:13 2020

@author: jay
"""
import pickle
import numpy as np
import csv
import datetime

#data = np.load('./5-celebrity-faces-embeddings.npz')
#trainnames = data['arr_1']
#all_names=list(set(trainnames))


def record_data(list_attendance):
#with open("./list_attendance1.pkl", 'rb') as file:
 #   list_attendance = pickle.load(file)
    rows=[]
    for names in list_attendance:
        curr=names.split("_")
        curr.append(datetime.datetime.now().ctime())
        rows.append(curr) 
        
        
    print(rows)
    
    with open("class_records.csv", 'a') as csvfile:  
        # creating a csv writer object  
        csvwriter = csv.writer(csvfile)  
            
        # writing the fields  
            
        # writing the data rows  
        csvwriter.writerows(rows) 
#with open("pickle_model.pkl", 'rb') as file:
#    model = pickle.load(file)
