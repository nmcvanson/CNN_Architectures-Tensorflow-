# disable tensorflow debugging messages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import h5py
import numpy as np
import tensorflow as tf
import math
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from resnet import ResNet
def load_dataset():
    train_dataset = h5py.File("train_signs.h5", "r")
    # your train set features
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])

    '''
    img = Image.fromarray(train_set_x_orig[im])
    for im in range(len(train_set_x_orig)):
        img = Image.fromarray(train_set_x_orig[im])
        #img = ImageEnhance.Contrast(img).enhance(1)
        img = ImageOps.equalize(img, mask = None)
        img = img.filter(ImageFilter.SMOOTH)
        train_set_x_orig[im] = np.array(img)
    train_set_x_orig2 = []
    for im in range(len(train_set_x_orig)):
        train_set_x_orig2.append(np.rot90(train_set_x_orig[im], axes=(-3, -2)))
    train_set_x_orig3 = []
    for im in range(len(train_set_x_orig)):
        train_set_x_orig3.append(np.rot90(train_set_x_orig2[im], axes=(-3, -2)))
    train_set_x_orig3 = np.array(train_set_x_orig3)
    train_set_x_orig4 = []
    for im in range(len(train_set_x_orig)):
        train_set_x_orig4.append(np.rot90(train_set_x_orig3[im], axes=(-3, -2)))
    train_set_x_orig4 = np.array(train_set_x_orig4)

    train_set_x_orig = np.concatenate((train_set_x_orig, train_set_x_orig2, train_set_x_orig3, train_set_x_orig4))
    '''
    
    train_set_y_orig = np.array(
        train_dataset["train_set_y"][:])  # your train set labels

    #train_set_y_orig = np.concatenate((train_set_y_orig, train_set_y_orig, train_set_y_orig, train_set_y_orig))

    
    test_dataset = h5py.File("test_signs.h5", "r")
    # your test set features
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])
        
    test_set_y_orig = np.array(
        test_dataset["test_set_y"][:])  # your test set labels

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y

def load_traffic_sign_data():
    data = np.load('./datasets/traffic_signs_dataset/Training/data.npy')
    labels = np.load('./datasets/traffic_signs_dataset/Training/target.npy')
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size = 0.2, random_state = 0)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape) 
    y_train = to_categorical(y_train,43) 
    y_test = to_categorical(y_test,43)
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    # test ResNet50 for 6-signs-dataset
    # model = ResNet(name = "ResNet50", input_shape = (64, 64, 3), classes = 6)
    # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

    # # Normalize image vectors
    # X_train = X_train_orig / 255.
    # X_test = X_test_orig / 255.

    # # Convert training and test labels to one hot matrices
    # Y_train = convert_to_one_hot(Y_train_orig, 6).T
    # Y_test = convert_to_one_hot(Y_test_orig, 6).T

    # print ("number of training examples = " + str(X_train.shape[0]))
    # print ("number of test examples = " + str(X_test.shape[0]))
    # print ("X_train shape: " + str(X_train.shape))
    # print ("Y_train shape: " + str(Y_train.shape))
    # print ("X_test shape: " + str(X_test.shape))
    # print ("Y_test shape: " + str(Y_test.shape))
    # model.fit(X_train, Y_train, epochs = 10, batch_size = 32)

    #for traffic_signs:
    model = ResNet(name = "ResNet50", input_shape = (64, 64, 3),training= True, classes = 43)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    X_train, X_test, y_train, y_test = load_traffic_sign_data()
    X_train = X_train/ 255.
    X_test = X_test/ 255.

    print ("number of training examples = " + str(X_train.shape[0]))
    print ("number of test examples = " + str(X_test.shape[0]))
    print ("X_train shape: " + str(X_train.shape))
    print ("Y_train shape: " + str(y_train.shape))
    print ("X_test shape: " + str(X_test.shape))
    print ("Y_test shape: " + str(y_test.shape))
    model.fit(X_train, y_train, epochs = 20, batch_size = 32)
