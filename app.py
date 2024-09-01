from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import streamlit as st
import pickle


model = load_model(r'D:\Pycharm\Computer Vision Project\NLP Project\Fake News\model.h5')
tokenizer = pickle.load(open(r'D:\Pycharm\Computer Vision Project\NLP Project\Fake News\tokenizer.bin', 'rb'))
class_label = {1: 'Fake', 0: 'Real'}
st.title('Fake News Project.📰🗞️')

text = st.text_input('Enter THe Text')
if text:
    text = tokenizer.texts_to_sequences([text])
    pad_seq = np.array(pad_sequences(text, maxlen=13468, padding="pre", truncating="pre"))
    prediction = model.predict(pad_seq)

    if st.button('predict'):
        st.write(class_label[prediction.argmax()])