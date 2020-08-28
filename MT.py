import datetime
start=datetime.datetime.now() 

import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
from face_recognize import *

user='newuser'
# =============================================================================
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# 
def face_extractor(img):
#     
     gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
     faces = face_classifier.detectMultiScale(gray, 1.3, 5)
     
     if faces is():
         return None
     
     for(x,y,w,h) in faces:
         cropped_face = img[y:y+h, x:x+w]
         
     return cropped_face
     
     
#cap = cv2.VideoCapture(0)
#count = 0
 
# =============================================================================
#while True:
#     ret, frame = cap.read()
 #    if face_extractor(frame) is not None:
  #       count+=1
   #      face = cv2.resize(face_extractor(frame),(200,200))
    #     face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
         
#         file_name_path = 'cm_100_pic/'+str(user)+str(count)+'.jpg'
#         cv2.imwrite(file_name_path,face)
#         
#         cv2.putText(face,str(count),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
#         cv2.imshow('Face Cropper',face)
#     else:
 #        print("Face Not Found")
  #       pass     
     
#     if cv2.waitKey(1)==13 or count==100:
 #        break
     
#cap.release()
#cv2.destroyAllWindows()
#print('Collecting Samples Complete!!!')

# =============================================================================
# =============================================================================
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
# =============================================================================
# =============================================================================



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
             file_test_path = 'test_pic/'+str(user)+'.jpg'
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
             all_names = out_encoder.inverse_transform([0,1,2,3])
            
             #print('Predicted: %s (%.3f)' % (predict_names[0], class_probability))
             print('Predicted: \n%s \n%s' % (all_names, yhat_prob[0]*100))
             names=predict_names[0]
             cv2.putText(image,str(predict_names), (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
             cv2.imshow('Face Cropper', image)
                         
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

end=datetime.datetime.now() 