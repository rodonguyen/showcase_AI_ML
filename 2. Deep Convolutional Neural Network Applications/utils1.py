#
# Utility functions for CAB420, Assignment 1B, Q1
# Author: Simon Denman (s.denman@qut.edu.au)
#

from scipy.io import loadmat        # to load mat files
import matplotlib.pyplot as plt     # for plotting
import numpy as np                  # for reshaping, array manipulation
import cv2                          # for colour conversion
import tensorflow as tf             # for bulk image resize
from time import process_time
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
import numpy as np
import pandas as pd
from time import process_time
import cv2
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Activation
from tensorflow.keras.layers import AveragePooling2D, Input, Flatten
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow import keras
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from tensorflow.keras.utils import model_to_dot, plot_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import pickle
import time

def savedata(data, filename):
    with open(f"{filename}", "wb") as f:
        pickle.dump(data, f)


def loaddata(filename):
    with open(f"{filename}", "rb") as f:
        return pickle.load(f)

# Load data for Q1
#  train_path: path to training data mat file
#  test_path:  path to testing data mat file
#
#  returns:    arrays for training and testing X and Y data
#
def load_data(train_path, test_path):

    # load files
    train = loadmat(train_path)
    test = loadmat(test_path)

    # transpose, such that dimensions are (sample, width, height, channels), and divide by 255.0
    train_X = np.transpose(train['train_X'], (3, 0, 1, 2)) / 255.0
    train_Y = train['train_Y']
    # change labels '10' to '0' for compatability with keras/tf. The label '10' denotes the digit '0'
    train_Y[train_Y == 10] = 0
    train_Y = np.reshape(train_Y, -1)

    # transpose, such that dimensions are (sample, width, height, channels), and divide by 255.0
    test_X = np.transpose(test['test_X'], (3, 0, 1, 2)) / 255.0
    test_Y = test['test_Y']
    # change labels '10' to '0' for compatability with keras/tf. The label '10' denotes the digit '0'
    test_Y[test_Y == 10] = 0
    test_Y = np.reshape(test_Y, -1)

    # return loaded data
    return train_X, train_Y, test_X, test_Y

# vectorise an array of images, such that the shape is changed from {samples, width, height, channels} to
# (samples, width * height * channels)
#   images: array of images to vectorise
#
#   returns: vectorised array of images
#
def vectorise(images):
    # use numpy's reshape to vectorise the data
    return np.reshape(images, [len(images), -1])

# Plot some images and their labels. Will plot the first 100 samples in a 10x10 grid
#  x: array of images, of shape (samples, width, height, channels)
#  y: labels of the images
#
def plot_images(x, y):
    fig = plt.figure(figsize=[15, 18])
    for i in range(100):
        ax = fig.add_subplot(10, 10, i + 1)
        ax.imshow(x[i,:])
        ax.set_title(y[i])
        ax.axis('off')

# Resize an array of images
#  images:   array of images, of shape (samples, width, height, channels)
#  new_size: tuple of the new size, (new_width, new_height)
#
#  returns:  resized array of images, (samples, new_width, new_height, channels)
#
def resize(images, new_size):
    # tensorflow has an image resize funtion that can do this in bulk
    # note the conversion back to numpy after the resize
    return tf.image.resize(images, new_size).numpy()
          
# Convert images to grayscale
#   images:  array of colour images to convert, of size (samples, width, height, 3)
#
#   returns: array of converted images, of size (samples, width, height, 1)
#
def convert_to_grayscale(images):
    # storage for converted images
    gray = []
    # loop through images
    for i in range(len(images)):
        # convert each image using openCV
        gray.append(cv2.cvtColor(images[i,:], cv2.COLOR_BGR2GRAY))
    # pack converted list as an array and return
    return np.array(gray)  



def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
    """2D Convolution-Batch Normalization-Activation stack builder

    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)

    # Returns
        x (tensor): tensor as input to the next layer
    """
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x


def resnet_v1(inputs, filters, num_res_blocks, pool_size):
    """ResNet Version 1 Model builder [a]

    Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
    Last ReLU is after the shortcut connection.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filters is
    doubled. Within each stage, the layers have the same number filters and the
    same number of filters.

    # Arguments
        inputs (layer):         the input tensor
        filters ([int]):        number of filters in each stage, length of list determines number of stages
        num_res_blocks (int):   number of residual blocks per stage
        pool_size (int):        size of the average pooling at the end

    # Returns
        output after global average pooling and flatten, ready for output
    """
    x = resnet_layer(inputs=inputs,
                     num_filters=filters[0])

    # Instantiate the stack of residual units
    for stack, filters in enumerate(filters):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2  # downsample
            y = resnet_layer(inputs=x,
                             num_filters=filters,
                             strides=strides)
            y = resnet_layer(inputs=y,
                             num_filters=filters,
                             activation=None)
            if stack > 0 and res_block == 0:  # first layer but not first stack
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=filters,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = keras.layers.add([x, y])
            x = Activation('relu')(x)
 
    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    x = AveragePooling2D(pool_size=pool_size)(x)
    y = Flatten()(x)

    return y


def resnet_v2(inputs, filters, num_res_blocks, pool_size):
    """ResNet Version 2 Model builder [b]

    Stacks of (1 x 1)-(3 x 3)-(1 x 1) BN-ReLU-Conv2D or also known as
    bottleneck layer
    First shortcut connection per layer is 1 x 1 Conv2D.
    Second and onwards shortcut connection is identity.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filter maps is
    doubled. Within each stage, the layers have the same number filters and the
    same filter map sizes.

    # Arguments
        inputs (layer):         the input tensor
        filters ([int]):        number of filters in each stage, length of list determines number of stages
        num_res_blocks (int):   number of residual blocks per stage
        pool_size (int):        size of the average pooling at the end

    # Returns
        output after global average pooling and flatten, ready for output
    """

    x = resnet_layer(inputs=inputs,
                     num_filters=filters[0],
                     conv_first=True)

    # Instantiate the stack of residual units
    for stage, filters in enumerate(filters):
        num_filters_in = filters
        for res_block in range(num_res_blocks):
            activation = 'relu'
            batch_normalization = True
            strides = 1
            if stage == 0:
                num_filters_out = num_filters_in * 4
                if res_block == 0:  # first layer and first stage
                    activation = None
                    batch_normalization = False
            else:
                num_filters_out = num_filters_in * 2
                if res_block == 0:  # first layer but not first stage
                    strides = 2    # downsample

            # bottleneck residual unit
            y = resnet_layer(inputs=x,
                             num_filters=num_filters_in,
                             kernel_size=1,
                             strides=strides,
                             activation=activation,
                             batch_normalization=batch_normalization,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_in,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_out,
                             kernel_size=1,
                             conv_first=False)
            if res_block == 0:
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters_out,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = keras.layers.add([x, y])

        num_filters_in = num_filters_out

    # Add classifier on top.
    # v2 has BN-ReLU before Pooling
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = AveragePooling2D(pool_size=pool_size)(x)
    x = Flatten()(x)
    # x = layers.Dense(256, activation='relu')(x)
    # x = layers.Dropout(0.5)(x)
    # x = layers.Dense(64, activation='relu')(x)
    return x


def train_and_eval(model, x_train, y_train, x_val, y_val, x_test, y_test, filename, batch_size, epochs, monitor='val_accuracy', patience=10):
    
    checkpoint = ModelCheckpoint(filename, verbose=1, monitor=monitor, save_best_only=True, mode='auto')
    earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=patience)

    time1 = time.time()
    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_data=(x_val, y_val),
                        callbacks=[earlystopping, checkpoint])
    time2 = time.time()
    savedata(history.history, filename[:-3]+'_history')
    # Load the best saved model based on the criteria in 'monitor'
    model.load_weights(filename)
    model.save(filename)

    # Other metrics report
    classification_report_ = classification_report(y_test, tf.argmax(model.predict(x_test), axis=1))
    savedata(classification_report_, filename[:-3]+'_classification_report')
    print(classification_report_)


    print('Training time:', time2-time1, 's')
    pred = model.predict(x_train);
    indexes = tf.argmax(pred, axis=1)
    print('Training Accuracy:', np.sum(indexes == y_train[:,0]) / len(y_train))
    
    
    # IMPORTANT FIGURES
    fig = plt.figure(figsize=[30, 20])
    
    ax = fig.add_subplot(2, 3, 1)
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['accuracy'], label='train_accuracy')
    ax.set_xlabel('Number of epochs')
    ax.set_ylabel('Value')
    ax.legend()
    ax.set_title('Training Performance')

    ax = fig.add_subplot(2, 3, 2)
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    ax.set_xlabel('Number of epochs')
    ax.set_ylabel('Value')
    ax.legend()
    ax.set_title('Validation Performance')


    ax = fig.add_subplot(2, 3, 3)
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    ax.set_xlabel('Number of epochs')
    ax.set_ylabel('Value')
    ax.legend()
    ax.set_title('Still Validation Performance')


    
    ax = fig.add_subplot(2,3,4)
    pred = model.predict(x_train);
    indexes = tf.argmax(pred, axis=1)
    cm = confusion_matrix(y_train, indexes)
    c = ConfusionMatrixDisplay(cm, display_labels=range(10))
    c.plot(ax = ax)    
    ax.set_title('Training Result')
    
    ax = fig.add_subplot(2,3,5)
    pred = model.predict(x_val);
    indexes = tf.argmax(pred, axis=1)
    print('Validation Accuracy:', np.sum(indexes == y_val[:,0]) / len(y_val))       ###
    cm = confusion_matrix(y_val, indexes)
    c = ConfusionMatrixDisplay(cm, display_labels=range(10))
    c.plot(ax = ax)
    ax.set_title('Validation Result')


    ax = fig.add_subplot(2,3,6)
    time1 = process_time()
    pred = model.predict(x_test);
    indexes = tf.argmax(pred, axis=1)
    print('Test Accuracy:', np.sum(indexes == y_test[:,0]) / len(y_test))           ###
    print('Inference Time:', process_time()-time1, 's')                             ###
    cm = confusion_matrix(y_test, indexes)
    c = ConfusionMatrixDisplay(cm, display_labels=range(10))
    c.plot(ax = ax)
    ax.set_title('Testing Result')


    # ax = fig.add_subplot(3,2,5)
    # pred = model.predict(x_train);
    # indexes = tf.argmax(pred, axis=1)
    # cm = confusion_matrix(y_train, indexes)
    # hm = sns.heatmap(cm/np.sum(cm), annot=True, 
    #             fmt='.2%', cmap='Blues')
    # hm.plot(ax=ax)
    # ax.set_title('Training result\n');
    # ax.set_xlabel('\nPredicted')
    # ax.set_ylabel('Actual');
    # ## Ticket labels - List must be in alphabetical order
    # ax.xaxis.set_ticklabels(range(10))
    # ax.yaxis.set_ticklabels(range(10))


    # ax = fig.add_subplot(3,2,6)
    # pred = model.predict(x_test);
    # indexes = tf.argmax(pred, axis=1)
    # cm = confusion_matrix(y_test, indexes)
    # hm = sns.heatmap(cm/np.sum(cm), annot=True, 
    #             fmt='.2%', cmap='Blues')
    # hm.plot(ax=ax)
    # ax.set_title('Testing result\n');
    # ax.set_xlabel('\nPredicted')
    # ax.set_ylabel('Actual');
    # ## Ticket labels - List must be in alphabetical order
    # ax.xaxis.set_ticklabels(range(10))
    # ax.yaxis.set_ticklabels(range(10))

    plt.savefig(filename[:-3])


def my_model(inputs):
    # our model, input in an image shape
    # inputs = keras.Input(shape=(32, 32, 1,), name='img')
    x = inputs
    # x = layers.Reshape((-1,32,32), input_shape=(32,32,1,))(x)
    # run pairs of conv layers, all 3s3 kernels
    x = layers.Conv2D(filters=8, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = layers.Conv2D(filters=8, kernel_size=(3, 3), padding='same', activation=None)(x)
    # batch normalisation, before the non-linearity
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    # spatial dropout, this will drop whole kernels, i.e. 20% of our 3x3 filters will be dropped out rather
    # than dropping out 20% of the invidual pixels
    x = layers.SpatialDropout2D(0.2)(x)
    # max pooling, 2x2, which will downsample the image
    x = layers.MaxPool2D(pool_size=(2, 2))(x)
    # rinse and repeat with 2D convs, batch norm, dropout and max pool
    x = layers.Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = layers.Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation=None)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.SpatialDropout2D(0.2)(x)
    x = layers.MaxPool2D(pool_size=(2, 2))(x)
    # final conv2d, batch norm and spatial dropout
    x = layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation=None)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.SpatialDropout2D(0.2)(x)

    # flatten layer
    x = layers.Flatten()(x)
    # we'll use a couple of dense layers here, mainly so that we can show what another dropout layer looks like
    # in the middle
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(64, activation='relu')(x)
    
    return x