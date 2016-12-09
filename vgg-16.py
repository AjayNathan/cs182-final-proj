# Important reference: https://arxiv.org/pdf/1509.01626v3.pdf

from keras.models import Graph
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.regularizers import l2, activity_l2
from keras.constraints import maxnorm
import cv2, numpy as np
import os, os.path
import pandas as pd
from sklearn.cross_validation import train_test_split as sk_split
from keras.utils import np_utils

def characterModel(weights_path = None):
    model = Graph()

    # input
    model.add_input(name='image', input_shape=(140,63,1))

    # convolution layers
    model.add_node(Convolution2D(16, 1, 1, activation='relu'), name='c1', input='image')
    
    # two fully-connected layers
    model.add_node(Flatten(), name='f', input='c1')
    model.add_node(Dense(2048, activation='relu'), name='d1', input='f')
    model.add_node(Dropout(0.5), name='dr1', input='d1')
    model.add_node(Dense(2, activation='softmax'), name='d2', input='dr1')

    # output
    model.add_output(name='output', input='d2')

    # load weights, if there is a path to weights file
    if weights_path:
        model.load_weights(weights_path)

    return model

def characterModel2(weights_path = None):
    model = Graph()

    # input
    model.add_input(name='image', input_shape=(140,63,1))

    # convolution layers
    model.add_node(Convolution2D(32, 3, 3, border_mode='same', activation='relu', W_constraint=maxnorm(3)), name='c1', input='image')
    model.add_node(Dropout(0.2), name='dr1', input='c1')
    model.add_node(Convolution2D(32, 3, 3, border_mode='same', activation='relu', W_constraint=maxnorm(3)), name='c2', input='dr1')
    model.add_node(MaxPooling2D((2,2)), name='mp', input='c2')

    # fully-connected layers
    model.add_node(Flatten(), name='f', input='mp')
    model.add_node(Dense(512, activation='relu'), name='d1', input='f')
    model.add_node(Dropout(0.5), name='dr2', input='d1')
    model.add_node(Dense(2, activation='softmax'), name='d2', input='dr2')

    # output
    model.add_output(name='output', input='d2')

    # load weights, if there is a path to weights file
    if weights_path:
        model.load_weights(weights_path)

    return model

def characterModel3(weights_path = None):
    model = Graph()

    # input
    model.add_input(name='image', input_shape=(140,63,1))

    # convolution layers
    model.add_node(Convolution2D(32, 3, 3, border_mode='same', activation='relu'), name='c1', input='image')
    model.add_node(Dropout(0.2), name='dr1', input='c1')
    model.add_node(Convolution2D(32, 3, 3, border_mode='same', activation='relu'), name='c2', input='dr1')
    model.add_node(MaxPooling2D((2,2)), name='mp1', input='c2')

    model.add_node(Convolution2D(64, 3, 3, border_mode='same', activation='relu'), name='c3', input='mp1')
    model.add_node(Dropout(0.2), name='dr2', input='c3')
    model.add_node(Convolution2D(64, 3, 3, border_mode='same', activation='relu'), name='c4', input='dr2')
    model.add_node(MaxPooling2D((2,2)), name='mp2', input='c4')

    model.add_node(Convolution2D(128, 3, 3, border_mode='same', activation='relu'), name='c5', input='mp2')
    model.add_node(Dropout(0.2), name='dr3', input='c5')
    model.add_node(Convolution2D(128, 3, 3, border_mode='same', activation='relu'), name='c6', input='dr3')
    model.add_node(MaxPooling2D((2,2)), name='mp3', input='c6')

    # fully-connected layers
    model.add_node(Flatten(), name='f', input='mp3')
    model.add_node(Dropout(0.2), name='dr4', input='f')
    model.add_node(Dense(1024, activation='relu'), name='d1', input='dr4')
    model.add_node(Dropout(0.2), name='dr5', input='d1')
    model.add_node(Dense(512, activation='relu'), name='d2', input='dr5')
    model.add_node(Dropout(0.2), name='dr6', input='d2')
    model.add_node(Dense(2, activation='softmax'), name='d3', input='dr6')

    # output
    model.add_output(name='output', input='d2')

    # load weights, if there is a path to weights file
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
    data = pd.read_csv('datasets/tweets.csv', delimiter=',')

    c_data = data[data.handle == "HillaryClinton"]["text"].as_matrix()

    t_data = data[data.handle == "realDonaldTrump"]["text"].as_matrix()

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

    X_train, X_test, y_train, y_test = sk_split(x_data, y_data, test_size=0.10, random_state=41)

    Y_train = np_utils.to_categorical(y_train)
    Y_test = np_utils.to_categorical(y_test)

    return X_train, X_test, Y_train, Y_test, y_test

if __name__ == '__main__':
    batch_size = 32
    nb_epoch = 5
    lr = 0.01
    decay = lr/float(nb_epoch)

    # process data
    X_train, X_test, Y_train, Y_test, y_test = processData()

    # load model from weights and compile
    model = characterModel3()
    sgd = SGD(lr=lr, momentum=0.9, decay=decay, nesterov=False)
    model.compile(optimizer=sgd, loss={'output': 'categorical_crossentropy'}, metrics=['accuracy'])

    print model.summary()

    # train model and save weights
    # training = 3 epochs * 31s per epoch on Tesla M40 GPU
    # testing loss = 0.0982
    model.fit({'image': X_train, 'output': Y_train}, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, validation_data={'image': X_test, 'output': Y_test})
    model.save_weights('weights4.h5')

    print model.evaluate({'image': X_test, 'output': Y_test}, verbose=0)

    predictions = model.predict({"image": np.asarray(X_test)}, verbose=0)

    diffs = []
    for i, pred in enumerate(predictions["output"]):
        if pred[y_test[i]] < pred[y_test[i] ^ 1]:
            diffs.append(pred[y_test[i] ^ 1] - pred[y_test[i]])
    print diffs, len(diffs), len(predictions["output"])
    print np.mean(np.asarray(diffs))
