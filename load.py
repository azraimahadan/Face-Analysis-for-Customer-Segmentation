import numpy as np
import streamlit as st
import tensorflow as tf
from keras.models import load_model
import tensorflow as tf
from keras.layers import TFSMLayer

@st.cache_resource
def init(): 
    #loaded_model = load_model('model.h5')

    model_layer = TFSMLayer('saved_model/my_model', call_endpoint='serving_default')

    class CustomModel(tf.keras.Model):
        def __init__(self, model_layer):
            super(CustomModel, self).__init__()
            self.model_layer = model_layer
        
        def call(self, inputs):
            return self.model_layer(inputs)

    # Create an instance of the custom model
    loaded_model = CustomModel(model_layer)
    return loaded_model