# cancer-streamlit
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/alliwene/cancer-streamlit/main/cancer-class.py)

Trained and deployed an hybrid pretrained EfficientNetB6 and Bidirectional LSTM model to predict whether the lesion in given image is benign or malignant. Dataset used was obtained from The International Skin Imaging Collaboration (ISIC) 2020 challenge 
[dataset](https://www.kaggle.com/c/siim-isic-melanoma-classification/data).

Notebook for EDA, CNN and CNN+BiLSTM model training [notebook](https://www.kaggle.com/alliwene/cancer-class/notebook), tflite model conversion [notebook](notebooks/convert_cancer_model.ipynb), Python [script](cancer-class.py) to build web app. 