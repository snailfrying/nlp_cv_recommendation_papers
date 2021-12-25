# -*- coding: utf-8 -*-
# @File : MobileNet.py
# @Author: snailfrying
# @Time : 2021/9/12 10:30
# @Software: PyCharm

import numpy as np

from tensorflow import keras
from tensorflow.keras import layers, utils
from tensorflow.python.keras import backend
from tensorflow.keras.preprocessing import image
from tensorflow.python.keras.engine import training
from tensorflow.keras.applications import MobileNet, MobileNetV2

# Model / data parameters
num_classes = 10
inp = (224, 224)
input_shape = (224, 224, 3)

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

def keras_MobileNet(input_shape, classes, alaph=0.5,depth_multiplier=1,):
    model = MobileNet(
        input_shape=input_shape, classes=classes, alpha=alaph, depth_multiplier=depth_multiplier, weights=None)
    return model

# conv / s2
def conv_block(inputs, filters, alpha, kernel=(3, 3), strides=(1, 1)):
    channel_axis = 1 if backend.image_data_format() == 'channels_first' else -1

    filters = int(filters * alpha)
    x = layers.Conv2D(
        filters,
        kernel,
        padding='same',
        use_bias=False,
        strides=strides,
        name='conv1')(inputs)
    x = layers.BatchNormalization(axis=channel_axis, name='conv1_bn')(x)
    x = layers.ReLU(6., name='conv1_relu')(x)
    return x

# conv dw / s*
def depthwise_conv_block(inputs,
                         pointwise_conv_filters,
                         alpha,
                         depth_multiplier=1,
                         strides=(1, 1),
                         block_id=1):
    channel_axis = 1 if backend.image_data_format() == 'channels_first' else -1

    pointwise_conv_filters = int(pointwise_conv_filters * alpha)

    if strides == (1, 1):
        x = inputs
    else:
        x = layers.ZeroPadding2D(((0, 1), (0, 1)), name='conv_pad_%d' % block_id)(
            inputs)

    '''
    具体看论文、源码
    output[b, i, j, k * channel_multiplier + q] = sum_{di, dj}
    filter[di, dj, k, q] * input[b, strides[1] * i + rate[0] * di,
                           strides[2] * j + rate[1] * dj, k]
    '''
    x = layers.DepthwiseConv2D((3, 3),
                               padding='same' if strides == (1, 1) else 'valid',
                               depth_multiplier=depth_multiplier,
                               strides=strides,
                               use_bias=False,
                               name='conv_dw_%d' % block_id)(
        x)
    x = layers.BatchNormalization(
        axis=channel_axis, name='conv_dw_%d_bn' % block_id)(
        x)
    x = layers.ReLU(6., name='conv_dw_%d_relu' % block_id)(x)

    x = layers.Conv2D(
        pointwise_conv_filters, (1, 1),
        padding='same',
        use_bias=False,
        strides=(1, 1),
        name='conv_pw_%d' % block_id)(
        x)
    x = layers.BatchNormalization(
        axis=channel_axis, name='conv_pw_%d_bn' % block_id)(
        x)
    x = layers.ReLU(6., name='conv_pw_%d_relu' % block_id)(x)
    return x


def MobileNet_ym(input_shape, classes, alpha=0.5, depth_multiplier=1):
    image_inputs = keras.Input(input_shape)

    x = conv_block(image_inputs, 32, alpha, strides=(2, 2))
    x = depthwise_conv_block(x, 64, alpha, depth_multiplier, block_id=1)

    x = depthwise_conv_block(
        x, 128, alpha, depth_multiplier, strides=(2, 2), block_id=2)
    x = depthwise_conv_block(x, 128, alpha, depth_multiplier, block_id=3)

    x = depthwise_conv_block(
        x, 256, alpha, depth_multiplier, strides=(2, 2), block_id=4)
    x = depthwise_conv_block(x, 256, alpha, depth_multiplier, block_id=5)

    x = depthwise_conv_block(
        x, 512, alpha, depth_multiplier, strides=(2, 2), block_id=6)
    x = depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=7)
    x = depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=8)
    x = depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=9)
    x = depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=10)
    x = depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=11)

    x = depthwise_conv_block(
        x, 1024, alpha, depth_multiplier, strides=(2, 2), block_id=12)
    x = depthwise_conv_block(x, 1024, alpha, depth_multiplier, block_id=13)

    if backend.image_data_format() == 'channels_first':
        shape = (int(1024 * alpha), 1, 1)
    else:
        shape = (1, 1, int(1024 * alpha))

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Reshape(shape, name='reshape_1')(x)
    x = layers.Dropout(0.5, name='dropout')(x)
    x = layers.Conv2D(classes, (1, 1), padding='same', name='conv_preds')(x)
    x = layers.Reshape((classes,), name='reshape_2')(x)
    x = layers.Activation(activation='softmax',
                          name='predictions')(x)
    model = training.Model(image_inputs, x, name='MobileNet')
    return model


if __name__ == '__main__':
    x_train, y_train, x_test, y_test = read_data()

    model = MobileNet_ym(input_shape=input_shape, classes=num_classes, alpha=0.5, depth_multiplier=1)
    # 编译模型
    opt = keras.optimizers.Adam(learning_rate=1e-3, epsilon=1e-7)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    model.fit(
        x=x_train, y=y_train, batch_size=10, epochs=2, validation_split=0.2, validation_data=[x_test, y_test])
    model.summary()

    pre = model.evaluate(x_test, y_test, batch_size=500)
    print('test_loss: ', pre[0], 'test_acc: ', pre[1])
