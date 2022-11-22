import cv2 
import os 
import streamlit as st
import tensorflow  as tf
from PIL import Image
from tensorflow.keras.models import load_model
import numpy as np
model  = load_model(os.path.join('models','happysad.h5'))

def load_image(image_file):
	img = Image.open(image_file)
	return img


st.title("Deep Image Classifier")
st.subheader("Image")
image_file = st.file_uploader("Upload Images", type=["png","jpg","jpeg"])
if image_file is not None:
    st.image(load_image(image_file),width=500)
    press = st.button("Classify")
    if press :
        imgs = Image.open(image_file)
        imgs.save('img1.jpg')
        img= cv2.imread('img1.jpg')
        

        resize = tf.image.resize(img, (256,256))
        ypred = model.predict(np.expand_dims(resize/255, 0))
        if ypred > 0.5:
            st.header('Sad' )
        else:
            st.header('Happy')




