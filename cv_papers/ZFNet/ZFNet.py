# -*- coding: utf-8 -*-
# @File : ZFNEt.py
# @Author: snailfrying
# @Time : 2021/9/12 10:30
# @Software: PyCharm

import numpy as np

from tensorflow import keras
from tensorflow.keras import layers, utils
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

# ZFNet 主要作用是对AlexNet进行内部可视化，
# 以及进行调参优化。
def ZFNet(input_shape, classes):
    img_input = keras.Input(shape=input_shape, name='inputs')
    x = layers.Conv2D(filters=96, kernel_size=(7, 7), strides=(2, 2), activation='relu')(img_input)
    x = layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters=256, kernel_size=(5, 5), strides=(2, 2), activation='relu', padding="same")(x)
    x = layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same")(x)
    x = layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2))(x)

    x = layers.Flatten()(x)
    x = layers.Dense(4096, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(4096, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(classes, activation='softmax')(x)

    model = training.Model(img_input, x, name='ZFNet')
    return model


if __name__ == '__main__':
    x_train, y_train, x_test, y_test = read_data()

    model = ZFNet(input_shape=input_shape, classes=num_classes)
    # 编译模型
    opt = keras.optimizers.Adam(learning_rate=1e-3, epsilon=1e-7)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x=x_train, y=y_train, batch_size=500, epochs=10, validation_split=0.2, validation_data=[x_test, y_test])
    model.summary()

    pre = model.evaluate(x_test, y_test, batch_size=500)
    print('test_loss: ', pre[0], 'test_acc: ', pre[1])
