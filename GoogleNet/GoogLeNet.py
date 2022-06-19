import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    MaxPooling2D,
    AveragePooling2D,
    Dropout,
    Dense,
    Activation
)
from tensorflow.keras import Model
from block import *
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def GoogLeNet(input_shape, classes):
    """
    Arguments:
    input_shape: shape of input image
    classes: number of classes

    Return:
    model: a Model() instance in Keras
    """
    X_input = Input(input_shape)
    #conv block
    X = conv_block(X_input, filters = 64, kernel_size = 7, stride = 2, padding = "same")

    #max pool
    X = MaxPooling2D(pool_size = (3,3), strides = (2,2), padding = 'same')(X)

    #conv block
    X = conv_block(X, filters = 64, kernel_size = 1, stride = 1, padding = 'valid')

    #conv block
    X = conv_block(X, filters = 192, kernel_size = 3, stride = 1, padding = 'same')
    
    #max pool
    X = MaxPooling2D(pool_size = (3,3), strides = (2,2), padding = 'same')(X)

    #inception (3a)
    X = inception_block(X, filters_1x1 = 64, filters_pre3x3 = 96, filters_3x3 = 128, filters_pre5x5 = 16, filters_5x5 = 32, filters_after_maxpool = 32)

    #inception (3b)
    X = inception_block(X, filters_1x1 = 128, filters_pre3x3 = 128, filters_3x3 = 192, filters_pre5x5 = 32, filters_5x5 = 96, filters_after_maxpool = 64)

    #max pool
    X = MaxPooling2D(pool_size = (3,3), strides = (2,2), padding = 'same')(X)

    #inception (4a)
    X = inception_block(X, filters_1x1 = 192, filters_pre3x3 = 96, filters_3x3 = 208, filters_pre5x5 = 16, filters_5x5 = 48, filters_after_maxpool = 64)

    #first auxiliary softmax classifier
    auxiliary1 = auxiliary_block(X, classes = classes)

    #inception (4b)
    X = inception_block(X, filters_1x1 = 160, filters_pre3x3 = 112, filters_3x3 = 224, filters_pre5x5 = 24, filters_5x5 = 64, filters_after_maxpool = 64)

    #inception (4c)
    X = inception_block(X, filters_1x1 = 128, filters_pre3x3 = 128, filters_3x3 = 256, filters_pre5x5 = 24, filters_5x5 = 64, filters_after_maxpool = 64)

    #inception (4d)
    X = inception_block(X, filters_1x1 = 112, filters_pre3x3 = 144, filters_3x3 = 288, filters_pre5x5 = 32, filters_5x5 = 64, filters_after_maxpool = 64)

    #second auxiliary softmax classifier
    auxiliary2 = auxiliary_block(X, classes = classes)

    #inception (4e)
    X = inception_block(X, filters_1x1 = 256, filters_pre3x3 = 160, filters_3x3 = 320, filters_pre5x5 = 32, filters_5x5 = 128, filters_after_maxpool = 128)

    # max pool
    X = MaxPooling2D(pool_size = (3,3), strides = (2,2), padding = 'same')(X)

    #inception (5a)
    X = inception_block(X, filters_1x1 = 256, filters_pre3x3 = 160, filters_3x3 = 320, filters_pre5x5 = 32, filters_5x5 = 128, filters_after_maxpool = 128)

    #inception (5b)
    X = inception_block(X, filters_1x1 = 384, filters_pre3x3 = 192, filters_3x3 = 284, filters_pre5x5 = 48, filters_5x5 = 128, filters_after_maxpool = 128)

    #average pool

    X = AveragePooling2D(pool_size = (7,7), strides = (1,1), padding = 'valid')(X)

    #dropout
    X = Dropout(0.4)(X)

    X = Dense(units = classes)(X)
    X = Activation('softmax')(X)

    model = Model(inputs = X_input, outputs = [X, auxiliary1, auxiliary2], name = 'GoogLeNet_Inceptionv1')
    return model

if __name__ == "__main__":
    model = GoogLeNet(input_shape = (224, 224, 3), classes=1000)
    model.summary()

