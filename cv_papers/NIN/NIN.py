# -*- coding: utf-8 -*-
# @File : NIN.py
# @Author: snailfrying
# @Time : 2021/9/12 10:30
# @Software: PyCharm

import numpy as np

from tensorflow import keras
from tensorflow.keras import layers, utils
from tensorflow.python.keras import backend
from tensorflow.keras.preprocessing import image
from tensorflow.python.keras.engine import training

# Model / data parameters
num_classes = 10
input_shape = (224, 224, 3)



def read_data():
    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    # Scale images to the [0, 1] range
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    x_train = [image.smart_resize(img, (224, 224)) for img in x_train]
    x_test = [image.smart_resize(img, (224, 224)) for img in x_test]
    x_train, x_test = np.array(x_train), np.array(x_test)

    print("x_train shape:", x_train.shape)
    print(x_train.shape[0], "train samples")
    print(x_test.shape[0], "test samples")

    # convert class vectors to binary class matrices
    y_test = utils.to_categorical(y_test, num_classes)
    y_train = utils.to_categorical(y_train, num_classes)
    return x_train, y_train, x_test, y_test

def NIN(input_shape, classes):

    def mlp_layer(x, filters, kernel_size, strides, padding):
        x = layers.Conv2D(filters, kernel_size, strides=strides, padding=padding, activation='relu')(x)
        x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5)(x)
        x = layers.Conv2D(filters, kernel_size=1, strides=1, padding='same', activation='relu')(x)
        x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5)(x)
        x = layers.Conv2D(filters, kernel_size=1, strides=1, padding='same', activation='relu')(x)
        return x

    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

    image_inputs = keras.Input(input_shape)
    blocks = [96, 256, 384]
    # bloack 1
    x = mlp_layer(image_inputs, blocks[0], 11, 4, 'valid')
    x = layers.MaxPooling2D(pool_size=3, strides=2, padding='valid')(x)

    # bloack 2
    x = mlp_layer(x, blocks[1], 5, 1, 'same')
    x = layers.MaxPooling2D(pool_size=3, strides=2, padding='valid')(x)

    # bloack 3
    x = mlp_layer(x, blocks[2], 2, 1, 'same')
    x = layers.MaxPooling2D(pool_size=3, strides=2, padding='valid')(x)

    # bloack 4
    x = mlp_layer(x, classes, 2, 1, 'same')
    x = layers.GlobalAveragePooling2D(name='avg_pool')(x)

    model = training.Model(image_inputs, x, name='NIN')
    return model


if __name__ == '__main__':
    x_train, y_train, x_test, y_test = read_data()

    model = NIN(input_shape=input_shape, classes=num_classes)
    # 编译模型
    opt = keras.optimizers.Adam(learning_rate=1e-3, epsilon=1e-7)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    print(x_train.shape, y_train.shape,x_test.shape, y_test.shape)
    model.fit(x=x_train, y=y_train, batch_size=10, epochs=2, validation_split=0.2, validation_data=[x_test, y_test])
    model.summary()

    pre = model.evaluate(x_test, y_test, batch_size=500)
    print('test_loss: ', pre[0], 'test_acc: ', pre[1])
