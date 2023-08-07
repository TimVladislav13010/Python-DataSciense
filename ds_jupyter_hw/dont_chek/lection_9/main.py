import joblib
import streamlit as st
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.datasets import imdb

@st.cache_resource
def get_model():
    model = Sequential(
        [
            Dense(1024, activation="relu", input_shape=(10000,)),
            Dense(1024, activation="relu"),
            Dense(1024, activation="relu"),
            Dense(1, activation="sigmoid"),
        ]
    )

    model.load_weights("my_model.hdf5")
    return model


model = get_model()
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

text = st.text_input("Write text:")
if text:
    array = [word_index.get(word, -2) + 3 for word in text.split(" ")]
    new_vector = [0] * 10000
    for elem in array:
        st.write(elem)
        new_vector[elem] = 1

    new_vector = np.array(new_vector).reshape(1, 10000)

    st.write(new_vector)
    st.write(f"Prediction: {model.predict(new_vector)}")