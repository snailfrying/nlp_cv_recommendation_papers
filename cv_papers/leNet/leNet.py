# -*- coding: utf-8 -*-
# @File : LeNet.py
# @Author: snailfrying
# @Time : 2021/9/12 10:30
# @Software: PyCharm

import numpy as np

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras.engine import training

# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)


def read_data():
    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Scale images to the [0, 1] range
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    # Make sure images have shape (28, 28, 1)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    print("x_train shape:", x_train.shape)
    print(x_train.shape[0], "train samples")
    print(x_test.shape[0], "test samples")

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    return x_train, y_train, x_test, y_test


def LeNet(input_shape, classes):
    # 这里建立的应该是LeNet-5，即LeNet的升级版，原始版本用的sigmoid激活函数，以及没有dropout
    # 你可调试这两个组件，查看对模型的影响
    img_input = keras.Input(shape=input_shape, name='inputs')
    x = layers.Conv2D(filters=6, kernel_size=(5, 5), activation='relu')(img_input)  # sigmoid
    x = layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='same')(x)
    x = layers.Conv2D(filters=16, kernel_size=(5, 5), activation='relu')(x)
    x = layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='same')(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(units=120, activation='relu')(x)
    x = layers.Dense(units=84, activation='relu')(x)
    x = layers.Dense(units=classes, activation='softmax', name='predictions')(x)
    model = training.Model(img_input, x, name='LeNet')
    return model

if __name__ == '__main__':
    x_train, y_train, x_test, y_test = read_data()

    model = LeNet(input_shape=input_shape, classes=num_classes)
    # 编译模型
    opt = keras.optimizers.Adam(learning_rate=1e-3, epsilon=1e-7)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x=x_train, y=y_train, batch_size=500, epochs=10, validation_split=0.2, validation_data=[x_test, y_test])
    model.summary()

    pre = model.evaluate(x_test, y_test, batch_size=500)
    print('test_loss: ', pre[0], 'test_acc: ', pre[1])
