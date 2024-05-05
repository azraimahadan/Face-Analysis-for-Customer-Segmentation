import streamlit as st

import cv2
import numpy as np
from PIL import Image
from load import init
from model import Face
from tensorflow.keras.utils import img_to_array
from datetime import datetime
import pandas as pd
import os

# Initialize the model
loaded_model = init()
frame_number = 0

# Function to process the image and make predictions
def predict(img):
    global frame_number
    # Preprocess the image as needed
    # ...
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    #face_cascade = cv2.CascadeClassifier('haarcascade/haarcascade_frontalface_alt.xml')    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    frame_number+=1

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
                detected_face = cv2.resize(detected_face, (128,128), interpolation=cv2.INTER_AREA) #resize to 224x224
                
            except Exception as err:
                print("margin cannot be added (",str(err),")")
                detected_face = img[int(y):int(y+h), int(x):int(x+w)]
                detected_face = cv2.resize(detected_face, (128,128), interpolation=cv2.INTER_AREA)

            
            img_pixels = img_to_array(detected_face)
            img_pixels = np.expand_dims(img_pixels, axis = 0)
            img_pixels = img_pixels / 255

            prediction = loaded_model.predict(img_pixels)
            print(prediction)

            # Accessing the predicted values
            ethnicity_output = prediction['ethnicity_output']
            gender_output = prediction['gender_output']
            age_output = prediction['age_output']

            # Extracting specific values
            ethnicity_probabilities = ethnicity_output[0]
            gender_probability = gender_output[0][0]
            predicted_age = int(round(age_output[0][0]))

            # Finding the predicted ethnicity
            ethnicity_labels = ["White", "Black", "Asian", "Indian", "Others"]
            predicted_ethnicity_index = np.argmax(ethnicity_probabilities)
            predicted_ethnicity = ethnicity_labels[predicted_ethnicity_index]

            # Converting gender probability to a human-readable format
            predicted_gender = "Male" if gender_probability < 0.4 else "Female"

            # Constructing the final predictions dictionary
            predictions = {
                "predicted_text": "Face Detected",
                "age": predicted_age,
                "gender": predicted_gender,
                "ethnicity": f"{predicted_ethnicity}: {round(ethnicity_probabilities[predicted_ethnicity_index] * 100, 2)}%",
                "face_box": result_list
            }
            proba = np.round(100*ethnicity_probabilities[predicted_ethnicity_index], 2)
            if proba >= 30:
                if frame_number % 30 == 0 and proba >= 30:
                    frame_number = 0
                    df_prediction_file = pd.read_csv('prediction_data.csv', index_col=False)

                    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    data = {'time':[current_time], 'predicted_age': [predicted_age],'predicted_gender': [round(gender_probability)],'predicted_ethnicity': [predicted_ethnicity_index]}
                    df_new_prediction = pd.DataFrame(data)
                    df_prediction_file = pd.concat([df_prediction_file, df_new_prediction], ignore_index=True)
                    df_prediction_file.to_csv('prediction_data.csv', index=False)
                    print("Analysis Updated\n")
            
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

                age = f"Age: {predictions['age']}"
                gender = f"Gender: {predictions['gender']}"
                ethnicity = f"Ethnicity: {predictions['ethnicity']}"

                # Draw age
                cv2.putText(frame, age, (int(x + w + 25), int(y - 12)), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 51, 51), 1)
                # Draw gender
                cv2.putText(frame, gender, (int(x + w + 25), int(y + 10)), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 51, 51), 1)
                # Draw ethnicity
                cv2.putText(frame, ethnicity, (int(x + w + 25), int(y + 32)), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 51, 51), 1)



                cv2.rectangle(frame,(x-margin_x, y-margin_y),(x+w+margin_x, y+h+margin_y),(255, 51, 51),1)

        frame_placeholder.image(frame,channels="RGB")

        # Update the predictions dynamically
        age_placeholder.text("Predicted Age: {}".format(predictions["age"]))
        gender_placeholder.text("Predicted Gender: {}".format(predictions["gender"]))
        ethnicity_placeholder.text("Predicted Ethnicity: {}".format(predictions["ethnicity"]))


        key = cv2.waitKey(1)
        if key == ord('q') or stop_button:
            start_button = False  # Break the loop if 'q' is pressed or stop button is clicked

    # Release the video capture object
    cap.release()
    #cv2.destroyAllWindows()  # Close all OpenCV windows

# Run the Streamlit app
if __name__ == "__main__":
    main()