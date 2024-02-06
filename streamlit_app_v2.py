import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer
import queue

import cv2
import numpy as np
from PIL import Image

# Import your model initialization function and other necessary functions
from load import init
from model import Face
from tensorflow.keras.utils import img_to_array
import os
import av
from sample_utils.download import download_file
from sample_utils.turn import get_ice_servers

# Initialize the model
loaded_model = init()
result_queue: "queue.Queue[List[Detection]]" = queue.Queue()

def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")

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

            label = predictions['age']+"\n"+predictions['gender']+"\n"+predictions['ethnicity']
            #label = f"Predicted Age: {predictions['age']}\nPredicted Gender: {predictions['gender']}\nPredicted Ethnicity: {predictions['ethnicity']}"

            cv2.rectangle(img,(x-margin_x, y-margin_y),(x+w+margin_x, y+h+margin_y),(255, 51, 51),1)
            cv2.putText(img, label, (int(x+w+25), int(y-12)), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 51, 51), 1)


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
    result_queue.put(predictions)

    return av.VideoFrame.from_ndarray(img, format="bgr24")


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

    # Use the webrtc_streamer function to capture the video stream
    webrtc_ctx = webrtc_streamer(
        key="object-detection",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration={
            "iceServers": get_ice_servers(),
            "iceTransportPolicy": "relay",
        },
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)
    if st.checkbox("Show the detected labels", value=True):
        if webrtc_ctx.state.playing:
            labels_placeholder = st.empty()
            # NOTE: The video transformation with object detection and
            # this loop displaying the result labels are running
            # in different threads asynchronously.
            # Then the rendered video frames and the labels displayed here
            # are not strictly synchronized.
            while True:
                predictions = result_queue.get()
                # Update the predictions dynamically
                age_placeholder.text("Predicted Age: {}".format(predictions["age"]))
                gender_placeholder.text("Predicted Gender: {}".format(predictions["gender"]))
                ethnicity_placeholder.text("Predicted Ethnicity: {}".format(predictions["ethnicity"]))



# Run the Streamlit app
if __name__ == "__main__":
    main()