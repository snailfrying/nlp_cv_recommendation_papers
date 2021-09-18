# -*- coding: utf-8 -*-
# @File : Inception_v1.py
# @Author: snailfrying
# @Time : 2021/9/12 10:30
# @Software: PyCharm

import numpy as np

from tensorflow import keras
from tensorflow.keras import layers, utils
from tensorflow.python.keras import backend
from tensorflow.keras.preprocessing import image
from tensorflow.python.keras.engine import training
from tensorflow.keras.applications import InceptionV3, InceptionResNetV2

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


def inception_v3(input_shape, classes):
    model = InceptionV3(input_shape=input_shape, classes=classes, weights=None)
    return model


def inception_v1(input_shape, classes):
    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

    def inception_module(x, params, kernel_size=1, strides=1, padding='same'):
        param_1, param_2, param_3, param_4 = params
        # 1x1
        branch_1_1x1 = layers.Conv2D(filters=param_1[0], kernel_size=1, strides=1, padding=padding)(x)
        # 1x1->3x3
        branch_2_1x1 = layers.Conv2D(filters=param_2[0], kernel_size=1, strides=1, padding=padding)(x)
        branch_2_3x3 = layers.Conv2D(filters=param_2[1], kernel_size=3, strides=1, padding=padding)(branch_2_1x1)
        # 1x1->5x5
        branch_3_1x1 = layers.Conv2D(filters=param_3[0], kernel_size=1, strides=1, padding=padding)(x)
        branch_3_5x5 = layers.Conv2D(filters=param_3[1], kernel_size=5, strides=1, padding=padding)(branch_3_1x1)
        # 3x3->1x1
        branch_4_3x3 = layers.MaxPooling2D(pool_size=3, strides=1, padding=padding)(x)
        branch_4_1x1 = layers.Conv2D(filters=param_4[0], kernel_size=1, strides=1, padding=padding)(branch_4_3x3)

        x = layers.concatenate([branch_1_1x1, branch_2_3x3, branch_3_5x5, branch_4_1x1], axis=bn_axis)
        return x

    # define inception v1 networks modules filters parameters
    image_inputs = keras.Input(input_shape)
    params = [[(64,), (96, 128), (16, 32), (32,)],
              [(128,), (128, 192), (32, 96), (64,)],
              [(192,), (96, 208), (16, 48), (64,)],
              [(160,), (112, 224), (24, 64), (64,)],
              [(128,), (128, 256), (24, 64), (64,)],
              [(112,), (144, 288), (32, 64), (64,)],
              [(256,), (160, 320), (32, 128), (128,)],
              [(256,), (160, 320), (32, 128), (128,)],
              [(384,), (192, 384), (48, 128), (128,)]
              ]
    x = layers.Conv2D(filters=66, kernel_size=7, strides=2, padding='same')(image_inputs)
    x = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(x)
    x = layers.Conv2D(filters=64, kernel_size=2, strides=1, padding='same')(x)
    x = layers.Conv2D(filters=192, kernel_size=2, strides=1, padding='same')(x)
    x = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(x)
    # 3a
    x = inception_module(x, params=params[0])
    # 3b
    x = inception_module(x, params=params[1])
    x = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(x)
    # 4a
    x = inception_module(x, params=params[2])
    # 4b
    x = inception_module(x, params=params[3])
    # 4c
    x = inception_module(x, params=params[4])
    # 4d
    x = inception_module(x, params=params[5])
    # 4e
    x = inception_module(x, params=params[6])
    x = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(x)
    # 5a
    x = inception_module(x, params=params[7])
    # 5b
    x = inception_module(x, params=params[8])
    x = layers.AveragePooling2D(pool_size=7, strides=1, padding='valid')(x)

    x = layers.Flatten()(x)
    x = layers.Dropout(rate=0.4)(x)
    x = layers.Dense(units=classes, activation='linear')(x)
    x = layers.Dense(units=classes, activation='softmax')(x)

    model = training.Model(image_inputs, x, name='Inception')
    return model

if __name__ == '__main__':
    x_train, y_train, x_test, y_test = read_data()

    model = inception_v1(input_shape=input_shape, classes=num_classes)
    # 编译模型
    opt = keras.optimizers.Adam(learning_rate=1e-3, epsilon=1e-7)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    model.fit(x=x_train, y=y_train, batch_size=10, epochs=2, validation_split=0.2, validation_data=[x_test, y_test])
    model.summary()

    pre = model.evaluate(x_test, y_test, batch_size=500)
    print('test_loss: ', pre[0], 'test_acc: ', pre[1])
