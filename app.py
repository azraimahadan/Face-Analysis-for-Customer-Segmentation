from flask import Flask, render_template, url_for, Response, request
from model import Face
from load import *
import cv2, os
import numpy as np
from tensorflow.keras.utils import load_img, save_img, img_to_array
from keras.models import Model, Sequential, model_from_json
from PIL import Image

global model
pass_gender = 2
pass_age1 = pass_age2 = -1

loaded_model = init()

app = Flask(__name__)
face_cascade = cv2.CascadeClassifier('haarcascade/haarcascade_frontalface_alt.xml')
value = 0


@app.route('/', methods=["GET"])
def index():
    return render_template('index.html')

@app.route('/index2', methods=["GET"])
def index2():
    return render_template('index2.html')

@app.route('/index1_2', methods=["GET"])
def index1_2():
    return render_template('index1_2.html')


def livestream(source):
    while True:
        frame= source.input() #the final frame results to be displayed
        key = cv2.waitKey(1)
        #if key == ord('q'):
            #break
        if key == ord('p'):
            cv2.waitKey(-1)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def generate_frames():
    camera = cv2.VideoCapture(0)
    while True:
        ## read the camera frame
        success,frame=camera.read()
        if not success:
            break
        else:
            ret,buffer=cv2.imencode('.jpg',frame)
            frame=buffer.tobytes()

        yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed2')
def video_feed2():
    return Response(livestream(Face()), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/predict',methods=['GET'])
def predict():
    pass_gender = request.args.get('pass_gender')
    pass_race = request.args.get('pass_race')
    pass_age1 = request.args.get('pass_age1')
    pass_age2 = request.args.get('pass_age2')
    p_age = 0
    p_gender = ""
    p_race = ""
    races = ["White", "Asian", "Asian", "Indian","Others"]
    camera = cv2.VideoCapture(0)
    success,img=camera.read()
    
    if not success:
         return render_template('index2.html')
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        result_list2 = face_cascade.detectMultiScale(img, 1.3, 5)

        if len(result_list2) == 0:
             return render_template('index2.html',predicted_text="No Face Detected")   #if empty return index only
        
        else:
            max_w = max(result_list2,key=lambda item:item[2]) #find max w/biggest face
            for (x,y,w,h) in result_list2:   
                #add margin
                #margin_rate = 70
                try:
                    margin_x = int(w * 3 / 100)
                    margin_y = int(h * 10 / 100)+5
                    detected_face = img[int(y-margin_y):int(y+h+margin_y), int(x-margin_x):int(x+w+margin_x)]
                    detected_face = cv2.resize(detected_face, (120,120), interpolation=cv2.INTER_AREA) #resize to 224x224
                    
                except Exception as err:
                    print("margin cannot be added (",str(err),")")
                    detected_face = img[int(y):int(y+h), int(x):int(x+w)]
                    detected_face = cv2.resize(detected_face, (120,120), interpolation=cv2.INTER_AREA)

                
                img_pixels = img_to_array(detected_face)
                img_pixels = np.expand_dims(img_pixels, axis = 0)
                img_pixels = img_pixels / 255

                prediction = loaded_model.predict(img_pixels)
                print(prediction)

                aa= [int(i) for i in prediction[0].round().astype(int)]
                label_a=int("".join(str(i) for i in aa))
                gg = [int(i) for i in prediction[1].round().astype(int)]
                label_g=int("".join(str(i) for i in gg))
                ee = prediction[2].flatten()
                label_e = np.argmax(prediction[2])
                proba = np.round(100*prediction[2][0, label_e], 2)
                #print(ee)
                
                if False: # activate to dump
                    for i in range(0, len(races)):
                        if np.argmax(prediction_proba) == i:
                            print("* ", end='')
                        print(races[i], ": ", prediction_proba[0][i])
                    print("----------------")
                #find out age, gender and race
                p_race = races[label_e]
                    
                p_age = str(label_a)

                p_gender = label_g
                
                #Verify gender
                if pass_gender == str(p_gender):
                    predicted_text2 = "Gender Verified" + os.linesep
                elif pass_gender == str(2) or pass_gender==None:
                    predicted_text2 = ""
                else:
                    predicted_text2 = "Gender is not Verified" + os.linesep

                #print(pass_gender)
                #Verify race
                ind = np.argpartition(ee, -2)[-2:]
                p_race_second = races[ind[0]]
                proba_second = np.round(100*prediction[2][0, ind[0]], 2)
                #print(ind)
                if int(pass_race) == ind[0] or int(pass_race) == ind[1]:
                    predicted_text2 = predicted_text2 + "Ethnicity Verified" + os.linesep
                elif int(pass_race) == 5:
                    predicted_text2 = predicted_text2 + ""
                else:
                    predicted_text2 = predicted_text2 + "Ethnicity is not Verified"+ os.linesep
                
                #print(pass_race)

                #Verify Age
                #print(type(pass_age1))
                if int(pass_age1) != -1:
                    try:      
                        if int(pass_age1) <= label_a <= int(pass_age2):
                            predicted_text2 = predicted_text2 + "Age Verified"
                        elif int(pass_age1) == 0 and int(pass_age2) == 0:
                            predicted_text2 = predicted_text2 + " "
                        else:
                            predicted_text2 = predicted_text2 + "Age is not Verified"
                    
                    except:
                        print("TypeError")
                    
                else:
                    predicted_text2 = predicted_text2 + " "
                
                #print(pass_age1)
                print("Predicted text2: "+predicted_text2)
                if p_gender == 0: gender = "M"
                else: gender = "F"
                """for face in result_list2:
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
                    #detected_face = img[int(y):int(y+h), int(x):int(x+w)] #crop detected face
                #predicted_text2.headers["content-type"] = "text/plain"

                break
    
            return render_template('index2.html', predicted_text="Face Detected", \
                predicted_text2=predicted_text2,predicted_age = "Predicted age is "+ p_age,\
                    predicted_gender="Predicted gender is "+gender, \
                    predicted_ethnicity = "Predicted race is "+p_race+": "+str(proba)+"%, "+p_race_second+": "+str(proba_second)+"% ")

if __name__ == '__main__':
    app.static_folder = 'static'
    app.run(host="localhost", debug=True)