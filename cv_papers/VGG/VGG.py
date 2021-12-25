# -*- coding: utf-8 -*-
# @File : vgg.py
# @Author: snailfrying
# @Time : 2021/9/12 10:30
# @Software: PyCharm

import numpy as np

from tensorflow import keras
from tensorflow.keras import layers, utils
from tensorflow.keras.applications import vgg16
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

def vgg_16(input_shape, classes):
    model = vgg16.VGG16(input_shape=input_shape, classes=classes, weights=None)
    return model


def vgg_11_bn(input_shape, classes):
    # 原论文是LRN
    img_input = keras.Input(shape=input_shape, name='inputs')

    x = layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', name='block1_conv1')(img_input)
    x - layers.BatchNormalization()(x)
    x = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='block1_pool1')(x)

    x = layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu', name='block2_conv1')(x)
    x = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='block2_pool1')(x)

    x = layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu', name='block3_conv1')(x)
    x = layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu', name='block3_conv2')(x)
    x = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='block3_pool1')(x)

    x = layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu', name='block4_conv1')(x)
    x = layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu', name='block4_conv2')(x)
    x = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='block4_pool1')(x)

    x = layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu', name='block5_conv1')(x)
    x = layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu', name='block5_conv2')(x)
    x = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='block5_pool1')(x)

    x = layers.Flatten()(x)
    x = layers.Dense(4096, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(4096, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(1000, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(classes, activation='softmax')(x)

    model = training.Model(img_input, x, name='vgg13_bn')
    return model


if __name__ == '__main__':
    x_train, y_train, x_test, y_test = read_data()

    model = vgg_16(input_shape=input_shape, classes=num_classes)
    # 编译模型
    opt = keras.optimizers.Adam(learning_rate=1e-3, epsilon=1e-7)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    print(x_train.shape, y_train.shape,x_test.shape, y_test.shape)
    model.fit(x=x_train, y=y_train, batch_size=10, epochs=2, validation_split=0.2, validation_data=[x_test, y_test])
    model.summary()

    pre = model.evaluate(x_test, y_test, batch_size=500)
    print('test_loss: ', pre[0], 'test_acc: ', pre[1])
