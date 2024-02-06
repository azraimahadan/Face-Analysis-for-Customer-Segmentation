import streamlit as st

import cv2
import numpy as np
from PIL import Image

# Import your model initialization function and other necessary functions
from load import init
from model import Face
from tensorflow.keras.utils import img_to_array
import os

# Initialize the model
loaded_model = init()


# Function to process the image and make predictions
def predict(img):
    # Preprocess the image as needed
    # ...
    p_age = 0
    p_gender = ""
    p_race = ""
    races = ["White", "Asian", "Asian", "Indian","Others"]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #face_cascade = cv2.CascadeClassifier('haarcascade/haarcascade_frontalface_alt.xml')    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    result_list = face_cascade.detectMultiScale(img, 1.3, 5)
    if len(result_list) == 0:
        predictions = {
                "predicted_text": "Face Not Detected",
                "age": None,  # Include the predicted age directly
                "gender": None,  # Include the predicted gender directly
                "ethnicity": None,
                "face_box": None
            }
        return predictions

    else:
        max_w = max(result_list,key=lambda item:item[2]) #find max w/biggest face
        for (x,y,w,h) in result_list:   
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

            gender = label_g
            

            #print(pass_gender)
            #Verify race
            ind = np.argpartition(ee, -2)[-2:]
            p_race_second = races[ind[0]]
            proba_second = np.round(100*prediction[2][0, ind[0]], 2)
            
            #print(pass_race)

            #Verify Age
            #print(type(pass_age1))
            if gender == 0: p_gender = "M"
            else: p_gender = "F"

            predictions = {
                "predicted_text": "Face Detected",
                "age": p_age,  # Include the predicted age directly
                "gender": p_gender,  # Include the predicted gender directly
                "ethnicity": f"{p_race}: {proba}%",#, {p_race_second}: {proba_second}%"
                "face_box": result_list
            }
            
            break

    # Return the predictions
    return predictions

# Streamlit app
def main():
    st.title("Real Time Age, Gender and Race Detection")
    
    frame_placeholder = st.empty()
    age_placeholder = st.empty()
    gender_placeholder = st.empty()
    ethnicity_placeholder = st.empty()

    # Start and stop buttons
    start_button = st.button("Start")
    stop_button = st.button("Stop")
    # OpenCV video capture
    cap = cv2.VideoCapture(0)

    while cap.isOpened() and start_button:
        ret, frame = cap.read()
        if not ret:
            st.write("Video Capture Ended")
            break

        # Process the frame and make predictions
        predictions = predict(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if predictions["face_box"] is not None:
            for (x,y,w,h) in predictions["face_box"]:
                margin_x = int(w * 3 / 100)
                margin_y = int(h * 10 / 100)+5

                label = predictions['age']+"\n"+predictions['gender']+"\n"+predictions['ethnicity']
                #label = f"Predicted Age: {predictions['age']}\nPredicted Gender: {predictions['gender']}\nPredicted Ethnicity: {predictions['ethnicity']}"

                cv2.rectangle(frame,(x-margin_x, y-margin_y),(x+w+margin_x, y+h+margin_y),(255, 51, 51),1)
                cv2.putText(frame, label, (int(x+w+25), int(y-12)), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 51, 51), 1)

        frame_placeholder.image(frame,channels="RGB")

        # Update the predictions dynamically
        age_placeholder.text("Predicted Age: {}".format(predictions["age"]))
        gender_placeholder.text("Predicted Gender: {}".format(predictions["gender"]))
        ethnicity_placeholder.text("Predicted Ethnicity: {}".format(predictions["ethnicity"]))


        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or stop_button:
            start_button = False  # Break the loop if 'q' is pressed or stop button is clicked

    # Release the video capture object
    cap.release()
    cv2.destroyAllWindows()  # Close all OpenCV windows

# Run the Streamlit app
if __name__ == "__main__":
    main()