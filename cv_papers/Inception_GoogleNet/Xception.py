# -*- coding: utf-8 -*-
# @File : Xception.py
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
inp = (299, 299)
input_shape = (299, 299, 3)

def read_data():
    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    # Scale images to the [0, 1] range
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    x_train = [image.smart_resize(img, inp) for img in x_train]
    x_test = [image.smart_resize(img, inp) for img in x_test]
    x_train, x_test = np.array(x_train), np.array(x_test)

    print("x_train shape:", x_train.shape)
    print(x_train.shape[0], "train samples")
    print(x_test.shape[0], "test samples")

    # convert class vectors to binary class matrices
    y_test = utils.to_categorical(y_test, num_classes)
    y_train = utils.to_categorical(y_train, num_classes)
    return x_train, y_train, x_test, y_test

def Xception(input_shape, classes):

    bn_axis = -1 if backend.image_data_format() == 'channels_last' else 1

    image_inputs = keras.Input(input_shape)
    # Entry flow
    # block 1
    x = layers.Conv2D(filters=32, kernel_size=3, strides=2, name='block1_conv1')(image_inputs)
    x = layers.BatchNormalization(axis=bn_axis)(x)
    x = layers.Activation(activation='relu', name='block1_conv1_act')(x)

    x = layers.Conv2D(filters=64, kernel_size=3, name='block1_conv2')(x)
    x = layers.Activation(activation='relu', name='block1_conv2_act')(x)

    # block 2
    residual = layers.Conv2D(filters=128, kernel_size=1, strides=2, padding='same', name='block2_res')(x)

    '''
    具体看论文、源码
    output[b, i, j, k] = sum_{di, dj, q, r}
    input[b, strides[1] * i + di, strides[2] * j + dj, q] *
    depthwise_filter[di, dj, q, r] *
    pointwise_filter[0, 0, q * channel_multiplier + r, k]
    '''

    x = layers.SeparableConvolution2D(filters=128, kernel_size=3, padding='same', name='block2_sepconv1')(x)
    x = layers.Activation(activation='relu', name='block2_sepconv1_act')(x)
    x = layers.SeparableConvolution2D(filters=128, kernel_size=3, padding='same', name='block2_sepconv2')(x)
    x = layers.MaxPooling2D(pool_size=3, strides=2, padding='same', name='block2_maxpool')(x)
    x = layers.add([x, residual])

    # block 3
    residual = layers.Conv2D(filters=256, kernel_size=1, strides=2, padding='same', name='block3_res')(x)

    x = layers.Activation(activation='relu', name='block3_sepconv1_act')(x)
    x = layers.SeparableConvolution2D(filters=256, kernel_size=3, padding='same', name='block3_sepconv1')(x)
    x = layers.Activation(activation='relu', name='block3_sepconv2_act')(x)
    x = layers.SeparableConvolution2D(filters=256, kernel_size=3, padding='same', name='block3_sepconv2')(x)
    x = layers.MaxPooling2D(pool_size=3, strides=2, padding='same', name='block3_maxpool')(x)
    x = layers.add([x, residual])

    # block 4
    residual = layers.Conv2D(filters=728, kernel_size=1, strides=2, padding='same', name='block4_res')(x)

    x = layers.Activation(activation='relu', name='block4_sepconv1_act')(x)
    x = layers.SeparableConvolution2D(filters=728, kernel_size=3, padding='same', name='block4_sepconv1')(x)
    x = layers.Activation(activation='relu', name='block4_sepconv2_act')(x)
    x = layers.SeparableConvolution2D(filters=728, kernel_size=3, padding='same', name='block4_sepconv2')(x)
    x = layers.MaxPooling2D(pool_size=3, strides=2, padding='same', name='block4_maxpool')(x)
    x = layers.add([x, residual])

    # Middle flow
    # block 5-12
    for i in range(8):
        residual = x
        prefix = 'block' + str(i + 5)

        x = layers.Activation(activation='relu')(x)
        x = layers.SeparableConvolution2D(filters=728, kernel_size=3, padding='same', name=prefix + '_sepconv_v1')(x)

        x = layers.Activation(activation='relu')(x)
        x = layers.SeparableConvolution2D(filters=728, kernel_size=3, padding='same', name=prefix + '_sepconv_v2')(x)

        x = layers.Activation(activation='relu')(x)
        x = layers.SeparableConvolution2D(filters=728, kernel_size=3, padding='same', name=prefix + '_sepconv_v3')(x)

        x = layers.add([x, residual])

    # Exit Flow
    # block 13

    residual = layers.Conv2D(filters=1024, kernel_size=1, strides=2, padding='same', name='block13_res')(x)

    x = layers.Activation(activation='relu', name='block13_sepconv1_act')(x)
    x = layers.SeparableConvolution2D(filters=728, kernel_size=3, padding='same', name='block13_sepconv1')(x)

    x = layers.Activation(activation='relu', name='block13_sepconv2_act')(x)
    x = layers.SeparableConvolution2D(filters=1024, kernel_size=3, padding='same', name='block13_sepconv2')(x)
    x = layers.MaxPooling2D(pool_size=3, strides=2, padding='same', name='block13_maxpool')(x)
    x = layers.add([x, residual])

    # block 14
    x = layers.SeparableConvolution2D(filters=1536, kernel_size=3, padding='same', name='block14_sepconv1')(x)
    x = layers.Activation(activation='relu', name='block14_sepconv1_act')(x)
    x = layers.SeparableConvolution2D(filters=2048, kernel_size=3, padding='same', name='block14_sepconv2')(x)
    x = layers.Activation(activation='relu', name='block14_sepconv2_act')(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(classes, activation='softmax', name='prediction')(x)

    model = training.Model(image_inputs, x, name='Xception')
    return model


if __name__ == '__main__':
    x_train, y_train, x_test, y_test = read_data()

    model = Xception(input_shape=input_shape, classes=num_classes)
    # 编译模型
    opt = keras.optimizers.Adam(learning_rate=1e-3, epsilon=1e-7)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    model.fit(x=x_train, y=y_train, batch_size=10, epochs=2, validation_split=0.2, validation_data=[x_test, y_test])
    model.summary()

    pre = model.evaluate(x_test, y_test, batch_size=500)
    print('test_loss: ', pre[0], 'test_acc: ', pre[1])
