import cv2
#import imutils
#import config
from io import StringIO
import time
from time import gmtime, strftime
from datetime import datetime
import numpy as np
import pandas as pd
from tensorflow.keras.utils import load_img, save_img, img_to_array
from keras.models import Model, Sequential, model_from_json
from PIL import Image

from load import *

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
#from imutils.video import VideoStream

# Initialise runtime variables
# These variables are used to measure FPS rate
#start_time = time.time()

frame_number = 0
fps = 0
#face_mtcnn = MTCNN()
races = ["White", "Asian", "Asian", "Indian","Others"]
output_indexes = np.array([i for i in range(0, 101)])
enableGenderIcons =False
male_icon=""
female_icon=""



class Face(object):

    def __init__(self):
        # Initialise and load our pre-built DNN model
        #self.net = cv2.dnn.readNet("optimized_graph.pb", "optimized_graph.pbtxt")
        #self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
        #self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL_FP16)
        self.loaded_model = init()
        self.p_age=0
        self.p_gender=""
        self.p_race=""
        self.flag=True
        self.face_cascade = cv2.CascadeClassifier('haarcascade/haarcascade_frontalface_alt.xml')
        
        #self.mtcnn_detector = MTCNN()
        self.cap = cv2.VideoCapture(0) #webcam#VideoStream(src=0).start()
        # Warm-up model
        time.sleep(1)

    #def __del__(self):
    #    self.cap.stop()

    #def release(self):
    #    self.cap.stop()
        
    def input(self):

        # Access our global variables to track and calculate FPS
        global start_time
        global frame_number
        global fps

        # Initialize camera feed
        ret,img_colored = self.cap.read()
        
        img = img_colored
        img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        result_list = self.face_cascade.detectMultiScale(img2, 1.3, 5)
                # Measure the FPS rate every 30 frames
        
        frame_number += 1
        
        for result in result_list:
            x, y, w, h = result#['box']
            #x1, y1, x2, y2, width, height = self.face_localizer(result)
            if w > 50:
                
                detected_face = img2[int(y):int(y+h), int(x):int(x+w)] #crop detected face
                
                #add margin
                #margin_rate = 70
                try:
                    margin_x = int(w * 3 / 100)
                    margin_y = int(h * 10 / 100)+5
                    
                    detected_face = img2[int(y-margin_y):int(y+h+margin_y), int(x-margin_x):int(x+w+margin_x)]
                    detected_face = cv2.resize(detected_face,(120,120), interpolation=cv2.INTER_AREA) #resize to 224x224
                    
                    #display margin added face
                    #cv2.rectangle(img,(x-margin_x,y-margin_y),(x+w+margin_x,y+h+margin_y),(67, 67, 67),1)
                    
                except Exception as err:
                    #print("margin cannot be added (",str(err),")")
                    detected_face = img2[int(y):int(y+h), int(x):int(x+w)]
                    detected_face = cv2.resize(detected_face,(120,120), interpolation=cv2.INTER_AREA)
                
                #print("shape: ",detected_face.shape)
                
                #if detected_face.shape[0] > 0 and detected_face.shape[1] > 0 and detected_face.shape[2] >0: #sometimes shape becomes (264, 0, 3)
                    
                    
                img_pixels = img_to_array(detected_face)
                img_pixels = np.expand_dims(img_pixels, axis = 0)
                img_pixels = img_pixels / 255

                
                prediction = self.loaded_model.predict(img_pixels)
                
                print(prediction)
                aa= [int(i) for i in prediction[0].round().astype(int)]
                label_a=int("".join(str(i) for i in aa))
                gg = [int(i) for i in prediction[1].round().astype(int)]
                label_g=int("".join(str(i) for i in gg))
                #label_e = np.delete(prediction[2], np.argmax(prediction[2]))#.argsort()[:,-2]#np.argmax(prediction[2])
                #prediction[2][0,np.argmax(prediction[2])] = 0
                label_e = np.argmax(prediction[2])

                if False: # activate to dump
                    for i in range(0, len(races)):
                        if np.argmax(prediction_proba) == i:
                            print("* ", end='')
                        print(races[i], ": ", prediction_proba[0][i])
                    print("----------------")
                
                img = img_colored

                self.p_race = races[label_e]
                cv2.rectangle(img,(x-margin_x, y-margin_y),(x+w+margin_x, y+h+margin_y),(255, 51, 51),1) #draw rectangle to main image
                #cv2.rectangle(img,(left, top), (right, bottom),(255, 51, 51),1) #draw rectangle to main image
                #--------------------------
                #background
                overlay = img.copy()
                opacity = 0.4
                cv2.rectangle(img,(x+w+10,y-50),(x+w+170,y+15),(64,64,64),cv2.FILLED)
                cv2.addWeighted(overlay, opacity, img, 1 - opacity, 0, img)
                
                color = (255,255,255)
                proba = np.round(100*prediction[2][0, label_e], 2)
                if proba >= 10:
                    if frame_number % 50 == 0 and proba>=20:
                        frame_number = 0
                        df_prediction_file = pd.read_csv('prediction_data.csv', index_col=False)
                        if label_e==1:
                            label_e = 2
                        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        data = {'time':[current_time], 'predicted_age': [label_a],'predicted_gender': [label_g],'predicted_ethnicity': [label_e]}
                        df_new_prediction = pd.DataFrame(data)
                        df_prediction_file = df_prediction_file.append(df_new_prediction, ignore_index=True)
                        df_prediction_file.to_csv('prediction_data.csv', index=False)
                        print("---------------------------------------------------\n")
                        print(df_prediction_file)
                        print("---------------------------------------------------\n")
                        #print(df_new_prediction)
                    
                    label = str(self.p_race+" ("+str(proba)+"%)")
                    cv2.putText(img, label, (int(x+w+25), int(y-12)), cv2.FONT_HERSHEY_COMPLEX, 0.5, color, 1)
                        
                    #connect face and text
                    cv2.line(img,(int((x+x+w)/2),y+15),(x+w,y-20),(67, 67, 67),1)
                    cv2.line(img,(x+w,y-20),(x+w+10,y-20),(67, 67, 67),1)
                    
                    #find out age and gender
                    self.p_age = str(label_a)
                
                    self.p_gender = label_g
                
                    if self.p_gender == 0: gender = "M"
                    else: gender = "F"
            
                    #background for age gender declaration
                    info_box_color = (51,255,51)
                    #triangle_cnt = np.array( [(x+int(w/2), y+10), (x+int(w/2)-25, y-20), (x+int(w/2)+25, y-20)] )
                    triangle_cnt = np.array( [(x+int(w/2), y), (x+int(w/2)-20, y-20), (x+int(w/2)+20, y-20)] )
                    cv2.drawContours(img, [triangle_cnt], 0, info_box_color, -1)
                    #cv2.rectangle(img,(x+int(w/2)-50,y-20),(x+int(w/2)+50,y-90),info_box_color,cv2.FILLED)
                    center = x+int(w/2), y - 45
                    axes = 60, 30
                    angle = 180 
                    cv2.ellipse(img, center, axes, angle, 0, 360, info_box_color, cv2.FILLED)

                    #labels for age and gender
                    cv2.putText(img, self.p_age, (x+int(w/2), y - 35), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 111, 255), 2)
                
                    if enableGenderIcons:
                        if gender == 'M': gender_icon = male_icon
                        else: gender_icon = female_icon
                    
                        img[y-75:y-75+male_icon.shape[0], x+int(w/2)-45:x+int(w/2)-45+male_icon.shape[1]] = gender_icon
                    else:
                        cv2.putText(img, gender, (x+int(w/2)-42, y - 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 111, 255), 2)

        

        frame_text = str(frame_number)
            
        cv2.putText(img, frame_text,(5,30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,255,255), 1)
        """for face in result_list:
            x, y, w, h = face['box']
            if w>100: 
                center = [x+(w/2), y+(h/2)]
                max_border = max(w, h)
                
                # center alignment
                left = max(int(center[0]-(max_border/2)), 0)
                right = max(int(center[0]+(max_border/2)), 0)
                top = max(int(center[1]-(max_border/2)), 0)
                bottom = max(int(center[1]+(max_border/2)), 0)
                
                # crop the face
                center_img_k = img[top:top+max_border, 
                                left:left+max_border, :]
                center_img = np.array(Image.fromarray(center_img_k).resize([100,100]))
                center_img = cv2.cvtColor(center_img, cv2.COLOR_RGB2GRAY)"""
            
                    #cv2.imshow('frame', img)
        # Draw marker around detected faces
        """for i in range(0, predictions.shape[2]):

            confidence = predictions[0, 0, i, 2]

            # Filter out weak predictions by ensuring the `confidence level` is
            # no less than the minimum threshold set in config.py
            if confidence < 0.5:
                continue

            # Get the bounding box coordinates of the faces
            box = predictions[0, 0, i, 3:7] * \
                np.array([width, height, width, height])
            (startX, startY, endX, endY) = box.astype("int")

            # Draw bounding box and put confidence level on screen
            cv2.rectangle(img, (startX, startY),
                          (endX, endY), (255, 255, 0), 2)

            text = "{:.1f}%".format(confidence * 100)
            textY = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.putText(img, text, (startX, textY),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

        # Display FPS rate when available
        cv2.putText(img, "FPS: "+str(fps), (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 1)"""
        
        _, mjpeg = cv2.imencode('.jpg', img)

        #if self.p_age !=  and self.p_age != 
        return mjpeg.tobytes()