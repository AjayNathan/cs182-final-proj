# Important reference: https://arxiv.org/pdf/1509.01626v3.pdf

from keras.models import Graph
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
import cv2, numpy as np
import os, os.path
import pandas as pd
from sklearn.cross_validation import train_test_split as sk_split
from keras.utils import np_utils

def characterModel(weights_path = None):
    model = Graph()
    model.add_input(name='image', input_shape=(140,63,1))

    # convolution and pooling layer
    model.add_node(Convolution2D(32, 1, 1, activation='relu'), name='c1', input='image')
    model.add_node(MaxPooling2D((2,2), strides=(1,1)), name='mp1', input='c1')
    
    # two fully-connected layers
    model.add_node(Flatten(), name='f1', input='mp1')
    model.add_node(Dense(2048, activation='relu'), name='d1', input='f1')
    model.add_node(Dropout(0.5), name='dr1', input='d1')
    model.add_node(Dense(2, activation='softmax'), name='d3', input='dr1')

    model.add_output(name='output', input='d3')

    if weights_path:
        model.load_weights(weights_path)

    return model

def loadCharsFromTxt(text, dataset):
    image = np.zeros((140,63,1))
    
    # strip URLs
    words = text.split()
    i = 0
    while i < len(words): 
        if "t.co" in words[i]:
            del words[i]
        i += 1
    text = " ".join(words)
            
    charCount = 0
    for char in text:
        index = None
        if char.isalpha() and ord(char) >= ord('a'):
            index = ord(char) - ord('a') + 1
        elif char.isalpha() and ord(char) >= ord('A'):
            index = ord(char) - ord('A') + 37
        elif char.isdigit():
            index = ord(char) - ord('0') + 27
        elif char == " ":
            index = 0
                             
        if index:
            image[charCount, index][0] = 1
            charCount += 1
                    
    dataset.append(image)

def processData():
    data = (pd.read_csv('datasets/tweets.csv', delimiter=','), pd.read_csv('datasets/tweets2.csv', delimiter=','))

    c_data = np.concatenate((data[0][data[0].handle == "HillaryClinton"]["text"].as_matrix(), 
              data[1][data[0].handle == "HillaryClinton"]["text"].as_matrix()), axis=0)

    t_data = np.concatenate((data[0][data[0].handle == "realDonaldTrump"]["text"].as_matrix(), 
              data[1][data[0].handle == "realDonaldTrump"]["text"].as_matrix()), axis=0)

    clinton_tweets = []
    trump_tweets = []

    for text in c_data:
        loadCharsFromTxt(text, clinton_tweets)

    for text in t_data:
        loadCharsFromTxt(text, trump_tweets)
        
    clinton_y = np.zeros(len(clinton_tweets), dtype=int)
    trump_y = np.full(len(trump_tweets), 1, dtype=int)

    x_data = np.concatenate((clinton_tweets, trump_tweets), axis=0)
    y_data = np.concatenate((clinton_y, trump_y), axis=0)

    X_train, X_test, y_train, y_test = sk_split(x_data, y_data, test_size = 0.25, random_state = 42)

    print X_train.shape
    print y_train.shape

    Y_train = np_utils.to_categorical(y_train)
    Y_test = np_utils.to_categorical(y_test)

    return X_train, X_test, Y_train, Y_test

if __name__ == '__main__':
    batch_size = 128
    nb_epoch = 10

    # process data
    X_train, X_test, Y_train, Y_test = processData()

    # load model from weights and compile
    model = characterModel('my_weights.h5')
    model.compile(optimizer="adam", loss={'output': 'categorical_crossentropy'})

    # train model and save weights
    # training = 15 epochs * 67s per epoch on Tesla M40 GPU
    model.fit({'image': X_train, 'output': Y_train}, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, validation_data={'image': X_test, 'output': Y_test})
    model.save_weights('my_weights.h5')

    # evaluate model
    print model.evaluate({'image': X_test, 'output': Y_test}, verbose=0)

