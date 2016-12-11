# Important reference: https://arxiv.org/pdf/1509.01626v3.pdf

from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout, Activation, Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.constraints import maxnorm
from keras.utils import np_utils
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split as sk_split

def characterModel(weights_path = None):
    """smaller CNN"""
    model = Sequential()

    # convolution layers
    model.add(Convolution2D(16, 1, 1, input_shape=(140,63,1)))
    model.add(Activation('relu'))

    # two fully-connected layers
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2))
    model.add(Activation('softmax'))

    # load weights, if there is a path to weights file
    if weights_path:
        model.load_weights(weights_path)

    return model

def characterModel2(weights_path = None):
    """deeper CNN"""
    model = Sequential()

    # convolution layers
    model.add(Convolution2D(32, 3, 3, border_mode='same', W_constraint=maxnorm(3), input_shape=(140,63,1)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Convolution2D(32, 3, 3, border_mode='same', W_constraint=maxnorm(3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2,2)))

    # fully-connected layers
    model.add_node(Flatten())
    model.add_node(Dense(512, W_constraint=maxnorm(3)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2))
    model.add(Activation('softmax'))

    # load weights, if there is a path to weights file
    if weights_path:
        model.load_weights(weights_path)

    return model

def loadCharsFromTxt(text, dataset):
    """converts text to image-like matrix representation (input to CNN)"""
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
    """generates training and testing splits"""
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

    X_train, X_test, y_train, y_test = sk_split(x_data, y_data, test_size=0.10, random_state=42)

    Y_train = np_utils.to_categorical(y_train)
    Y_test = np_utils.to_categorical(y_test)

    return X_train, X_test, Y_train, Y_test

if __name__ == '__main__':
    # process data
    X_train, X_test, Y_train, Y_test = processData()

    # load model from weights and compile
    model1 = characterModel()
    model2 = characterModel2()

    model1.compile(optimizer='adam', loss={'output': 'categorical_crossentropy'}, metrics=['accuracy'])

    sgd = SGD(lr=0.01, momentum=0.9, decay=0.005, nesterov=False)
    model2.compile(optimizer=sgd, loss={'output': 'categorical_crossentropy'}, metrics=['accuracy'])

    print model1.summary()
    print model2.summary()

    # train models and save weights
    # training model 1 = 1 epochs * 14s per epoch on Tesla M40 GPU, overfits after 1 epoch
    # training model 2 = 8 epochs * 25s per epoch on Tesla M40 GPU, starts overfitting after ~8 epochs
    model1.fit(X_train, Y_train, batch_size=32, nb_epoch=1, verbose=1, validation_data=(X_test, Y_test))
    model1.save_weights('weights1.h5')

    model2.fit(X_train, Y_train, batch_size=32, nb_epoch=8, verbose=1, validation_data=(X_test, Y_test))
    model2.save_weights('weights2.h5')

    print model1.evaluate({'image': X_test, 'output': Y_test}, verbose=0)
    print model2.evaluate({'image': X_test, 'output': Y_test}, verbose=0)

    predictions1 = model1.predict({"image": np.asarray(X_test)}, verbose=0)
    predictions2 = model2.predict({"image": np.asarray(X_test)}, verbose=0)    
