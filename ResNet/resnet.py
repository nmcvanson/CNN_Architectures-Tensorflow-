import tensorflow as tf 
import numpy as np 
from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.initializers import random_uniform, glorot_uniform

def identity_block(input_tensor, kernel_size, filters,  ordinal_number, no_stage = 1,  training = True):
    """
    Arguments:
    input_tensor -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    kernel_size -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of intergers, defining the number of filters in the CONV layers of the main path
    training -- True: Behave in training mode
                False: behave in inference mode
    no_stage -- number of stage of block
    ordinal_number -- ordinal number of block
    Returns:
    X -- output of the identity block, tensor of shape (m, n_H, n_W, n_C)
    """
    filter1, filter2, filter3 = filters

    X_shortcut = input_tensor
    conv_name = "conv" + str(no_stage) + "_" + str(ordinal_number)
    bn_name = 'bn' + str(no_stage) + "_" + str(ordinal_number)
    #first component of main path
    X = Conv2D(filters = filter1 , kernel_size = 1, strides = (1,1), padding = 'valid', name = conv_name +"1_identity")(input_tensor)
    X = BatchNormalization(axis = 3, name = bn_name+ "1_identity")(X, training = training)
    X = Activation('relu')(X)

    #second component of main path
    X = Conv2D(filters = filter2, kernel_size = kernel_size, strides = (1,1), padding = 'same', name = conv_name+ "2_identity")(X)
    X = BatchNormalization(axis = 3, name = bn_name + "2_identity")(X, training = training)
    X = Activation('relu')(X)

    #third component of main path
    X = Conv2D(filters = filter3, kernel_size = 1, strides = (1,1), padding = 'valid', name = conv_name + "3_identity")(X)
    X = BatchNormalization(axis = 3, name = bn_name + "3_identity")(X, training = training)
    X = Activation('relu')(X)

    # skip-connection - add X_shortcut

    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X

def convolutional_block(input_tensor, kernel_size, filters, no_stage, strides = (2,2), training = True):
    """
    Arguments:
    input_tensor -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    kernel_size -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python lisst of intergers, defining the number of filters in the CONV layers of the main path
    strides -- python tuple of interger, specifying the stride to be used
    training -- True: Behave in training mode
                False: behave in inference mode
    Returns:
    X -- output of the identity block, tensor of shape (m, n_H, n_W, n_C)
    """
    filter1, filter2, filter3 = filters
    X_shortcut = input_tensor
    conv_name = "conv" + str(no_stage) + "_"
    bn_name = 'bn' + str(no_stage) + "_"
    #first component of main path
    X = Conv2D(filters = filter1, kernel_size = 1, strides = strides, padding = 'valid', name = conv_name +  "1_conv_block")(input_tensor)
    X = BatchNormalization(axis = 3, name = bn_name + "1")(X, training = training)
    X = Activation('relu')(X)

    #second component of main path
    X = Conv2D(filters = filter2, kernel_size = kernel_size, strides = (1,1), padding = 'same', name = conv_name + "2_conv_block")(X)
    X = BatchNormalization(axis = 3, name = bn_name + "2")(X, training = training)
    X = Activation('relu')(X)

    #third component of main path
    X = Conv2D(filters = filter3, kernel_size = 1, strides = (1,1), padding = 'same', name = conv_name + "3_conv_block")(X)
    X = BatchNormalization(axis = 3, name = bn_name + "3")(X, training = training)
    X = Activation('relu')(X)

    #shortcut path
    X_shortcut = Conv2D(filters = filter3, kernel_size = 1, strides = strides, padding = 'valid', name = conv_name + "shortcut")(X_shortcut)
    X_shortcut = BatchNormalization(axis = 3,  name = bn_name + "shortcut")(X_shortcut, training = training)
    
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X

def make_stage(input_tensor, no_blocks_in_stage, kernel_size, filters, strides,training, no_stage):
    """
    Arguments:
    input_tensor -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    no_blocks_in_stage -- number of blocks in stage
    kernel_size -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python lisst of intergers, defining the number of filters in the CONV layers of the main path
    strides -- python tuple of interger, specifying the stride to be used
    training -- True: Behave in training mode
                False: behave in inference mode
    Returns:
    X -- output of state, tensor of shape (m, n_H, n_W, n_C)
    """
    X = convolutional_block(input_tensor, kernel_size = kernel_size, filters = filters, strides= (1,1), training = training , no_stage = no_stage)
    for _ in range(no_blocks_in_stage -1):
        X = identity_block(X, kernel_size, filters, training= True, no_stage = no_stage, ordinal_number = _ + 1)
    return X

def ResNet(name = "ResNet50", input_shape = (64, 64, 3), training = True, classes = 6):
    """
    Arguments:
    name -- name of resnet architecture
    blocks_per_stage -- number of blocks per stage 2 - 5
    input_shape -- shape of the images of the dataset
    classes -- integer, number of classes

    Returns:
    model -- a Model() instance in Keras

    Model Architecture:
    Resnet50:
        CONV2D -> BATCHNORM -> RELU -> MAXPOOL  // conv1 -- stage1
            -> CONVBLOCK -> IDBLOCK * 2         // conv2_x -- stage2
            -> CONVBLOCK -> IDBLOCK * 3         // conv3_x -- stage3
            -> CONVBLOCK -> IDBLOCK * 5         // conv4_x -- stage4
            -> CONVBLOCK -> IDBLOCK * 2         // conv5_x -- stage5
            -> AVGPOOL
            -> TOPLAYER
    Resnet101:
        CONV2D -> BATCHNORM -> RELU -> MAXPOOL  // conv1 -- stage1
            -> CONVBLOCK -> IDBLOCK * 2         // conv2_x -- stage2
            -> CONVBLOCK -> IDBLOCK * 3         // conv3_x -- stage3
            -> CONVBLOCK -> IDBLOCK * 22        // conv4_x -- stage4
            -> CONVBLOCK -> IDBLOCK * 2         // conv5_x -- stage5
            -> AVGPOOL
            -> TOPLAYER
    Resnet152:
        CONV2D -> BATCHNORM -> RELU -> MAXPOOL  // conv1 -- stage1
            -> CONVBLOCK -> IDBLOCK * 2         // conv2_x -- stage2
            -> CONVBLOCK -> IDBLOCK * 7         // conv3_x -- stage3
            -> CONVBLOCK -> IDBLOCK * 35        // conv4_x -- stage4
            -> CONVBLOCK -> IDBLOCK * 2         // conv5_x -- stage5
            -> AVGPOOL
            -> TOPLAYER
    """
    
    if name == "ResNet50":
        blocks_per_stage = [3, 4, 6, 3]
    elif name == "ResNet101":
        blocks_per_stage = [3, 4, 23, 3]
    elif name == "ResNet152":
        blocks_per_stage = [3, 8, 36, 3]

    block2, block3, block4, block5 = blocks_per_stage
    X_input = Input(input_shape)
    #Zero-padding
    X = ZeroPadding2D((3,3))(X_input)

    #Stage 1:
    X = Conv2D(filters = 64, kernel_size = (7,7), strides = (2, 2), name = "conv1")(X)
    X = BatchNormalization(axis = 3, name = "bn_conv1")(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3,3), strides = (2,2))(X)

    #Stage 2:
    X = make_stage(X, no_blocks_in_stage = block2, kernel_size = 3, filters = [64,64,256], training = training, strides = 1, no_stage=2)
    
    #Stage 3:
    X = make_stage(X, no_blocks_in_stage = block3, kernel_size = 3, filters = [128,128,512], training = training, strides = 2, no_stage=3)

    #Stage 4:
    X = make_stage(X, no_blocks_in_stage = block4, kernel_size = 3, filters = [256, 256, 1024], training = training, strides = 2, no_stage=4)

    #Stage 5:
    X = make_stage(X, no_blocks_in_stage = block5, kernel_size = 3, filters = [512, 512, 2048], training = training, strides = 2, no_stage=5)

    X = AveragePooling2D(pool_size = (2,2))(X)

    X = Flatten()(X)
    X = Dense(classes, activation = 'softmax')(X)

    model = Model(inputs = X_input, outputs = X, name = name)
    return model



    