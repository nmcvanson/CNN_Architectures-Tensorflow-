from tensorflow.keras.layers import (
    Input,
    Conv2D,
    BatchNormalization,
    MaxPooling2D,
    Activation,
    Flatten,
    Dropout,
    Dense
)
from tensorflow.keras import Model
import tensorflow as tf 
import typing
#define architecture of VGG_network
#number represents number of output chanel after performing the CONV
#M - Maxpooling layer
VGG_network = {
    'VGG11': [64, 'M', 128, 'M', 256,256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
    'VGG19': [64, 64, 'M', 128, 128,' M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512]
}
def make_conv_maxpool_layers(X: tf.Tensor, architecture: typing.List[typing.Union[int, str]], activation: str = 'relu') -> tf.Tensor:
    """
    Arguments:
    X: input tensor
    architecture: number of output channel per conv layers
    activation: type of activation method

    Returns:
    X: output tensor
    """
    no_stage = 1
    conv_layer = 1

    for output in architecture:
        if type(output) == int:
            no_out_channels = output
            X = Conv2D(filters = no_out_channels, kernel_size = (3, 3), strides = (1,1) , padding = 'same', name = 'Conv' + str(no_stage) +'_' + str(conv_layer))(X)
            X = BatchNormalization(name = 'bn' + str(no_stage) + '_' + str(conv_layer))(X)
            X = Activation(activation)(X)
            conv_layer += 1
            
        else:
            X = MaxPooling2D(pool_size = (2,2), strides = (2,2), name = 'Maxpooling_' + str(no_stage))(X)
            no_stage += 1
            conv_layer = 1
    return X

def make_dense_layer(X: tf.Tensor, output_units: int, dropout = 0.5, activation = 'relu') -> tf.Tensor:
    """
    Arguments:
    X: input tensor
    output_units: number of output units
    dropout: dropout value for regularization
    activation: type of activation method

    Returns:
    X: output tensor
    """
    X = Dense(units = output_units)(X)
    X = BatchNormalization()(X)
    X = Activation(activation)(X)
    X = Dropout(dropout)(X)

    return X

def VGG(name: str, architecture: typing.List[typing.Union[int, str]], input_shape: typing.Tuple[int], classes: int= 1000) -> Model:
    """
    name - name of VGG network (VGG11, VGG13, VGG16, VGG19)
    architecture - architecture of vgg-x
    input_shape -- shapes of input images
    - classes -- number of classes
    """
    X_input = Input(input_shape)

    #conv and maxpooling layers
    X = make_conv_maxpool_layers(X_input, architecture)

    #flatten the output and make fully connected layers
    X = Flatten()(X)
    X = make_dense_layer(X, 4090)
    X = make_dense_layer(X, 4090)
    X = Dense(units = classes, activation = "softmax")(X)

    model = Model(inputs = X_input, outputs = X, name = name)
    return model

if __name__ == "__main__":
    model = VGG(name = "VGG-16", architecture = VGG_network["VGG16"], input_shape = (224,224,3), classes = 43)
    model.summary()

