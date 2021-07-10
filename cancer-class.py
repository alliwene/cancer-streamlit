import tensorflow as tf
from tensorflow.keras import preprocessing
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
fig = plt.figure()

st.write("""
# Melanoma Detection Using EfficientNet+BiLSTM

This web app predicts whether the lesion in given image is benign or malignant.
An Hybrid model which consists of a pretrained Convolutional Neural Network 
model (EfficientNetB6) and a Bidirectional Long Short Term Memory model 
(BiLSTM) was used for training. 

Dataset used was obtained from The International Skin Imaging Collaboration (ISIC) 2020 Challenge 
[Dataset](https://challenge2020.isic-archive.com/).

Upload jpeg image. 

""")

class Predict:
    def __init__(self, filename):
        self.img = self.get_image_from_upload()
        if self.img is not None:
            self.display_output()
            # self.pred = self.learn_inference(filename)
            self.get_prediction(filename)
    
    @staticmethod
    def get_image_from_upload():
        uploaded_file = st.file_uploader("Upload Files",type=['png','jpeg', 'jpg'])
        if uploaded_file is not None:
            return Image.open(uploaded_file)
        return None

    def display_output(self):
        plt.imshow(self.img)
        plt.axis("off")
        plt.title('Uploaded Image')
        st.pyplot(fig)

    def get_prediction(self, filename):

        if st.button('Classify'):
            # resize image and convert to tensor
            test_image = self.img.resize((512,512))
            test_image = preprocessing.image.img_to_array(test_image)
            test_image = test_image / 255.0
            test_image = np.expand_dims(test_image, axis=0)

            # Load the TFLite model and allocate tensors.
            interpreter = tf.lite.Interpreter(model_path=filename)
            interpreter.allocate_tensors()

            # Get input and output tensors.
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()

            # Test the model on image
            interpreter.set_tensor(input_details[0]['index'], test_image)
            interpreter.invoke()
            pred = interpreter.get_tensor(output_details[0]['index'])
            labels_dict = {0: 'Benign', 1: 'Malignant'}
            pred_class = (pred > 0.5).astype("int32")
            cancer_class = labels_dict[pred_class.item()]
            st.write(f'Prediction: {cancer_class}')
        else: 
            st.write(f'Click the button to classify') 

if __name__=='__main__':

    file_name = 'hybrid.tflite'

    predictor = Predict(file_name)
