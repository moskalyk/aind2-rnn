import numpy as np
from math import floor
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
import keras
import string

# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    # containers for input/output pairs
    X = []
    y = []

    for i in range(len(series) - window_size):
        slider_window = i + window_size
        output = slider_window + 1
        X.append(series[i:slider_window])
        y.append(series[slider_window:output])

    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)

    return X,y

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
    model = Sequential()
    model.add(LSTM(3, input_shape=(window_size, 1)))
    model.add(Dense(1, activation='tanh'))
    return model


### TODO: return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
    punctuation = ['!', ',', '.', ':', ';', '?']
    permitted_chars = list(string.ascii_lowercase) + punctuation
    unqiue_chars = set(text)

    for char in unqiue_chars:
        if char not in permitted_chars:
            text.replace(char, ' ')

    return text

### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []

    for i in range(len(text) - window_size):
        slider_window = i + window_size
        output = slider_window + 1
        inputs.append(text[i:slider_window])
        outputs.append(text[slider_window:output])

    return inputs,outputs

# TODO build the required RNN model: 
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):
    model = Sequential()
    model.add(LSTM(200, input_shape = (window_size, num_chars)))
    model.add(Dense(num_chars))
    model.add(Activation('softmax'))
    return model
