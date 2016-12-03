# model taken from https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3

from keras.models import Graph
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
import cv2, numpy as np
import os, os.path

def VGG_16(weights_path=None):
    model = Graph()
    model.add_input(name='image', input_shape=(3,224,224))
    model.add_node(ZeroPadding2D((1,1)), name='zp1', input='image')
    model.add_node(Convolution2D(64, 3, 3, activation='relu'), name='c1', input='zp1')
    model.add_node(ZeroPadding2D((1,1)), name='zp2', input='c1')
    model.add_node(Convolution2D(64, 3, 3, activation='relu'), name='c2', input='zp2')
    model.add_node(MaxPooling2D((2,2), strides=(2,2)), name='mp1', input='c2')

    model.add_node(ZeroPadding2D((1,1)), name='zp3', input='mp1')
    model.add_node(Convolution2D(128, 3, 3, activation='relu'), name='c3', input='zp3')
    model.add_node(ZeroPadding2D((1,1)), name='zp4', input='c3')
    model.add_node(Convolution2D(128, 3, 3, activation='relu'), name='c4', input='zp4')
    model.add_node(MaxPooling2D((2,2), strides=(2,2)), name='mp2', input='c4')

    model.add_node(ZeroPadding2D((1,1)), name='zp5', input='mp2')
    model.add_node(Convolution2D(256, 3, 3, activation='relu'), name='c5', input='zp5')
    model.add_node(ZeroPadding2D((1,1)), name='zp6', input='c5')
    model.add_node(Convolution2D(256, 3, 3, activation='relu'), name='c6', input='zp6')
    model.add_node(ZeroPadding2D((1,1)), name='zp7', input='c6')
    model.add_node(Convolution2D(256, 3, 3, activation='relu'), name='c7', input='zp7')
    model.add_node(MaxPooling2D((2,2), strides=(2,2)), name='mp3', input='c7')

    model.add_node(ZeroPadding2D((1,1)), name='zp8', input='mp3')
    model.add_node(Convolution2D(512, 3, 3, activation='relu'), name='c8', input='zp8')
    model.add_node(ZeroPadding2D((1,1)), name='zp9', input='c8')
    model.add_node(Convolution2D(512, 3, 3, activation='relu'), name='c9', input='zp9')
    model.add_node(ZeroPadding2D((1,1)), name='zp10', input='c9')
    model.add_node(Convolution2D(512, 3, 3, activation='relu'), name='c10', input='zp10')
    model.add_node(MaxPooling2D((2,2), strides=(2,2)), name='mp4', input='c10')

    model.add_node(ZeroPadding2D((1,1)), name='zp11', input='mp4')
    model.add_node(Convolution2D(512, 3, 3, activation='relu'), name='c11', input='zp11')
    model.add_node(ZeroPadding2D((1,1)), name='zp12', input='c11')
    model.add_node(Convolution2D(512, 3, 3, activation='relu'), name='c12', input='zp12')
    model.add_node(ZeroPadding2D((1,1)), name='zp13', input='c12')
    model.add_node(Convolution2D(512, 3, 3, activation='relu'), name='c13', input='zp13')
    model.add_node(MaxPooling2D((2,2), strides=(2,2)), name='mp5', input='c13')

    model.add_node(Flatten(), name='f1', input='mp5')
    model.add_node(Dense(4096, activation='relu'), name='d1', input='f1')
    model.add_node(Dropout(0.5), name='dr1', input='d1')
    model.add_node(Dense(4096, activation='relu'), name='d2', input='dr1')
    model.add_node(Dropout(0.5), name='dr2', input='d2')
    model.add_node(Dense(1000, activation='softmax'), name='d3', input='dr2')

    model.add_output(name='output', input='d3')

    if weights_path:
        model.load_weights(weights_path)

    return model

if __name__ == '__main__':
    img_dir = 'img'
    ex = ['jpg', 'jpeg', 'png']
    image_names = [im for im in os.listdir(img_dir) if any(e in im for e in ex)]
    num_images = len(image_names)

    images = range(num_images)
    for i in xrange(num_images):
        images[i] = cv2.resize(cv2.imread(img_dir + '/' + image_names[i]), (224, 224)).astype(np.float32)
        images[i][:,:,0] -= 103.939
        images[i][:,:,1] -= 116.779
        images[i][:,:,2] -= 123.68
        images[i] = images[i].transpose((2,0,1))
        images[i] = np.expand_dims(images[i], axis=0)

    # Test pretrained model
    model = VGG_16('vgg16_weights_graph.h5')
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss={'output': 'categorical_crossentropy'})

    out = map(lambda img: np.argmax(model.predict({'image': img})['output']) + 1, images)
    labels = ['']*num_images
    
    # go through synset_words.txt and assign labels to each image based on index in out
    with open('synset_words.txt') as f:
        for i, line in enumerate(f, 1):
            indices = [j for j, x in enumerate(out) if x == i]
            for k in xrange(len(indices)):
                labels[indices[k]] = line

    # print labels
    for i in xrange(num_images):
        if labels[i] != '':
            print image_names[i] + ': ' + labels[i]
        else:
            print image_names[i] + ': ' + 'None found\n'    
