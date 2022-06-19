import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    AveragePooling2D,
    Dropout,
    Flatten,
    Activation,
    Dense,
    BatchNormalization
)

def conv_block(input_tensor, filters, kernel_size, stride, padding):
    """
    Arguments:
    input_tensor: input tensor (m, n_H, n_W, channels)
    filters: integer - number of filter
    kernel_size: integer - specifying the shape of the middle conv's window for main path
    stride: integer - specifying the stride to be used
    padding: string - 'same' or 'valid'

    Return:
    X: output tensor (n_H, n_W, channels) (channels - number of new channels)
    """
    X = Conv2D(filters = filters, kernel_size = kernel_size, strides = (stride, stride), padding = padding)(input_tensor)
    #X = BatchNormalization()(X)
    #X = Activation("relu")(X)
    return X

def inception_block(input_tensor, filters_1x1, filters_pre3x3, filters_3x3, filters_pre5x5, filters_5x5, filters_after_maxpool):
    """
    Arguments:
    input_tensor: input tensor (m, n_H, n_W, channels)
    filters_1x1: number filters of 1x1 conv in first branch
    filters_pre3x3: number filters of 1x1 conv before 3x3 conv in second branch
    filters_3x3: number filters of 3x3 conv in second branch
    filters_pre5x5: number filters of 1x1 conv before 5x5 conv in third branch
    filters_5x5: number filters of 5x5 conv in third branch
    filters_after_maxpool: number filters of 1x1 conv after maxpooling in fourth branch

    Return:
    X: output tensor (n_H, n_W, channels) (channels = sum of channels after concatination)

    """
    #first branch
    conv_1x1 = conv_block(input_tensor, filters = filters_1x1, kernel_size = 1, stride = 1, padding = 'same')

    #second branch
    conv3x3 = conv_block(input_tensor, filters = filters_pre3x3, kernel_size = 1, stride = 1, padding = 'same')
    conv3x3 = conv_block(conv3x3, filters = filters_3x3, kernel_size = 3, stride = 1, padding = 'same')

    #third branch
    conv5x5 = conv_block(input_tensor, filters = filters_pre5x5, kernel_size = 1, stride = 1, padding = 'same')
    conv5x5 = conv_block(conv5x5, filters = filters_5x5, kernel_size = 5, stride = 1, padding = 'same')

    #fourth branch
    max_pooling = MaxPooling2D(pool_size = (3,3), strides = (1,1), padding = "same")(input_tensor)
    max_pooling = conv_block(max_pooling, filters = filters_after_maxpool, kernel_size = 1, stride = 1, padding = 'same')

    X = tf.keras.layers.Concatenate(axis = 3)([conv_1x1, conv3x3, conv5x5, max_pooling])
    return X

def auxiliary_block(input_tensor, classes):
    """
    Arguments:
    input_tensor: input tensor (m, n_H, n_W, channels)
    classes: integer - number of classes in auxiliary classifiers

    Return:
    X: output of the identity block, tensor of shape (n_H, n_W, filters)
    """
    X = AveragePooling2D(pool_size = (5,5), strides = (3,3), padding = 'valid')(input_tensor)
    X = conv_block(X, filters = 128, kernel_size = 1, stride = 1, padding = 'same')
    X = Flatten()(X)
    X = Dense(units = 1024)(X)
    X = Dropout(0.7)(X)
    X = Dense(units = classes)(X)
    X = Activation('softmax')(X)
    return X
