# model taken from https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3

from keras.models import Graph
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
import cv2, numpy as np
import os, os.path
import pandas as pd
from sklearn.cross_validation import train_test_split as sk_split
from keras.utils import np_utils

def VGG_16(weights_path=None):
    model = Graph()
    model.add_input(name='image', input_shape=(140,37,1))
    model.add_node(ZeroPadding2D((1,1)), name='zp1', input='image')
    model.add_node(Convolution2D(64, 1, 1, activation='relu'), name='c1', input='zp1')
    model.add_node(ZeroPadding2D((1,1)), name='zp2', input='c1')
    model.add_node(Convolution2D(64, 1, 1, activation='relu'), name='c2', input='zp2')
    model.add_node(MaxPooling2D((2,2), strides=(2,2)), name='mp1', input='c2')

    model.add_node(ZeroPadding2D((1,1)), name='zp3', input='mp1')
    model.add_node(Convolution2D(128, 1, 1, activation='relu'), name='c3', input='zp3')
    model.add_node(ZeroPadding2D((1,1)), name='zp4', input='c3')
    model.add_node(Convolution2D(128, 1, 1, activation='relu'), name='c4', input='zp4')
    model.add_node(MaxPooling2D((2,2), strides=(2,2)), name='mp2', input='c4')

    model.add_node(ZeroPadding2D((1,1)), name='zp5', input='mp2')
    model.add_node(Convolution2D(256, 1, 1, activation='relu'), name='c5', input='zp5')
    model.add_node(ZeroPadding2D((1,1)), name='zp6', input='c5')
    model.add_node(Convolution2D(256, 1, 1, activation='relu'), name='c6', input='zp6')
    model.add_node(ZeroPadding2D((1,1)), name='zp7', input='c6')
    model.add_node(Convolution2D(256, 1, 1, activation='relu'), name='c7', input='zp7')
    model.add_node(MaxPooling2D((2,2), strides=(2,2)), name='mp3', input='c7')

    model.add_node(ZeroPadding2D((1,1)), name='zp8', input='mp3')
    model.add_node(Convolution2D(512, 1, 1, activation='relu'), name='c8', input='zp8')
    model.add_node(ZeroPadding2D((1,1)), name='zp9', input='c8')
    model.add_node(Convolution2D(512, 1, 1, activation='relu'), name='c9', input='zp9')
    model.add_node(ZeroPadding2D((1,1)), name='zp10', input='c9')
    model.add_node(Convolution2D(512, 1, 1, activation='relu'), name='c10', input='zp10')
    model.add_node(MaxPooling2D((2,2), strides=(2,2)), name='mp4', input='c10')

    model.add_node(ZeroPadding2D((1,1)), name='zp11', input='mp4')
    model.add_node(Convolution2D(512, 1, 1, activation='relu'), name='c11', input='zp11')
    model.add_node(ZeroPadding2D((1,1)), name='zp12', input='c11')
    model.add_node(Convolution2D(512, 1, 1, activation='relu'), name='c12', input='zp12')
    model.add_node(ZeroPadding2D((1,1)), name='zp13', input='c12')
    model.add_node(Convolution2D(512, 1, 1, activation='relu'), name='c13', input='zp13')
    model.add_node(MaxPooling2D((2,2), strides=(2,2)), name='mp5', input='c13')

    model.add_node(Flatten(), name='f1', input='mp5')
    model.add_node(Dense(4096, activation='relu'), name='d1', input='f1')
    model.add_node(Dropout(0.5), name='dr1', input='d1')
    model.add_node(Dense(4096, activation='relu'), name='d2', input='dr1')
    model.add_node(Dropout(0.5), name='dr2', input='d2')
    model.add_node(Dense(2, activation='softmax'), name='d3', input='dr2')

    model.add_output(name='output', input='d3')

    if weights_path:
        model.load_weights(weights_path)

    return model

def model():
    model = Graph()
    model.add_input(name='image', input_shape=(140,37,1))
    #model.add_node(ZeroPadding2D((1,1)), name='zp', input='image')

    model.add_node(Convolution2D(512, 3, 3, activation='relu'), name='c1', input='image')
    model.add_node(MaxPooling2D((3,3), strides=(1,1)), name='mp1', input='c1')
    
    model.add_node(Convolution2D(512, 3, 3, activation='relu'), name='c2', input='mp1')
    model.add_node(MaxPooling2D((3,3), strides=(1,1)), name='mp2', input='c2')    

    model.add_node(Convolution2D(512, 3, 3, activation='relu'), name='c3', input='mp2')
    model.add_node(MaxPooling2D((3,3), strides=(1,1)), name='mp3', input='c3')    
    
    model.add_node(Convolution2D(512, 3, 3, activation='relu'), name='c4', input='mp3')
    model.add_node(MaxPooling2D((3,3), strides=(1,1)), name='mp4', input='c4')    
    
    model.add_node(Convolution2D(512, 3, 3, activation='relu'), name='c5', input='mp4')
    model.add_node(MaxPooling2D((3,3), strides=(1,1)), name='mp5', input='c5')  
    
    model.add_node(Convolution2D(512, 3, 3, activation='relu'), name='c6', input='mp5')
    model.add_node(MaxPooling2D((3,3), strides=(1,1)), name='mp6', input='c6')

    model.add_node(Flatten(), name='f1', input='mp6')
    model.add_node(Dense(2048, activation='relu'), name='d1', input='f1')
    model.add_node(Dropout(0.5), name='dr1', input='d1')
    model.add_node(Dense(2048, activation='relu'), name='d2', input='dr1')
    model.add_node(Dropout(0.5), name='dr2', input='d2')
    model.add_node(Dense(2, activation='softmax'), name='d3', input='dr2')

    model.add_output(name='output', input='d3')

    return model

def loadCharsFromTxt(text, dataset):
    image = np.zeros((140, 37, 1))
    words = text.lower().split()
    
    i = 0
    while i < len(words): 
        if "t.co" in words[i]:
            del words[i]
        i += 1
    
    text = " ".join(words)
            
    charCount = 0
    for char in text:
        index = None
        if char.isalpha():
            index = ord(char) - ord('a') + 1
        elif char.isdigit():
            index = ord(char) - ord('0') + 27
        elif char == " ":
            index = 0
                             
        if index:
            image[charCount, index][0] = 1
            charCount += 1
                    
    dataset.append(image)

if __name__ == '__main__':
    batch_size = 128
    nb_epoch = 10

    training_data = pd.read_csv('datasets/tweets.csv', delimiter=',')
    testing_data = pd.read_csv('datasets/tweets2.csv', delimiter=',')

    clinton_data_1 = training_data[training_data.handle == "HillaryClinton"]["text"].as_matrix()
    trump_data_1 = training_data[training_data.handle == "realDonaldTrump"]["text"].as_matrix()

    clinton_data_2 = testing_data[testing_data.handle == "HillaryClinton"]["text"].as_matrix()
    trump_data_2 = testing_data[testing_data.handle == "realDonaldTrump"]["text"].as_matrix()

    clinton_data = np.concatenate((clinton_data_1, clinton_data_2), axis=0)
    trump_data = np.concatenate((trump_data_1, trump_data_2), axis=0)

    clinton_dataset = []
    trump_dataset = []

    for text in clinton_data:
        loadCharsFromTxt(text, clinton_dataset)

    for text in trump_data:
        loadCharsFromTxt(text, trump_dataset)
        
    clinton_y = np.zeros(len(clinton_dataset))
    trump_y = np.full(len(trump_dataset), 1)

    x_data = np.concatenate((clinton_dataset, trump_dataset), axis=0)
    y_data = np.concatenate((clinton_y, trump_y), axis=0)

    X_train, X_test, y_train, y_test = sk_split(x_data, y_data, test_size = 0.25, random_state = 42)

    print X_test[0][0]

    Y_train = np_utils.to_categorical(y_train)
    Y_test = np_utils.to_categorical(y_test)

    model = model()
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    print "compiling"
    model.compile(optimizer="adam", loss={'output': 'categorical_crossentropy'})
    print "compiled"
    model.fit({'image': X_train, 'output': Y_train}, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, validation_data={'image': X_test, 'output': Y_test})

