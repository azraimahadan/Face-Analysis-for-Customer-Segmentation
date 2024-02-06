import numpy as np
import streamlit as st
from keras.models import model_from_json
import tensorflow as tf

@st.cache_resource
def init(): 
    json_file = open('weights/model2.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json,)
    # load weights into new model
    loaded_model.load_weights("weights/model2_weights.h5")
    print("Loaded model from disk")
    return loaded_model