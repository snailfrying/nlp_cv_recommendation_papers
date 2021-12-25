# -*- coding: utf-8 -*-
# @File : ResNetXt.py
# @Author: snailfrying
# @Time : 2021/9/22 10:30
# @Software: PyCharm

import numpy as np

from tensorflow import keras
from tensorflow.keras import layers, utils, regularizers
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
    # Make sure images have shape (28, 28, 1)
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


# 建立初始化层
def initial_conv_block(inputs, weight_decay=5e-4):
    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

    x = layers.Conv2D(64, (7, 7), strides=2, padding='same', use_bias=False)(inputs)
    x = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(x)
    x = layers.BatchNormalization(axis=bn_axis)(x)
    x = layers.Activation('relu')(x)

    return x


# 建立group convolution，即根据cardinality，把输入特征划分cardinality个group,然后对每一个group使用channels-filters提取特征，最后合并。
def group_conv(inputs, group_channels, cardinality, kernel_size=3, strides=1, padding='same', weight_decay=5e-4,
               name=None):
    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

    if cardinality == 1:
        x = layers.Conv2D(group_channels, kernel_size=kernel_size, strides=strides, padding=padding,
                          name=name + '_conv2')(inputs)
        x = layers.BatchNormalization(axis=bn_axis, name=name + '_conv2_bn')(x)
        x = layers.Activation(activation='relu', name=name + '_conv2_act')(x)
        return x

    feature_map_list = []
    for c in range(cardinality):
        x = layers.Lambda(
            lambda z: z[:, :, :, c * group_channels:(c + 1) * group_channels]
            if backend.image_data_format() == 'channels_last'
            else lambda z: z[:, c * group_channels:(c + 1) * group_channels, :, :]
        )(inputs)

        x = layers.Conv2D(filters=group_channels, kernel_size=kernel_size, strides=strides, padding=padding,
                          name=name + '_groupconv3_' + str(c))(x)
        feature_map_list.append(x)
    x = layers.concatenate(feature_map_list, axis=bn_axis, name=name + '_groupconv3_concat')
    x = layers.BatchNormalization(axis=bn_axis, name=name + '_groupconv3_bn')(x)
    x = layers.Activation(activation='relu', name=name + '_groupconv3_act')(x)
    return x


# 构建ResNetXt的block
def bottlencect_block(inputs, filters=64, cardinality=8, strides=1, weight_decay=5e-4, name=None):
    group_channels = filters // cardinality
    assert group_channels == 0
    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

    init = inputs
    if init.shape[-1] != 2 * filters:
        init = layers.Conv2D(filters * 2, (1, 1), padding='same', strides=(strides, strides), name=name + '_conv')(init)
        init = layers.BatchNormalization(axis=bn_axis, name=name + '_bn')(init)
    # conv 1*1
    x = layers.Conv2D(filters, kernel_size=1, strides=strides, padding='same', name=name + '_conv1')(inputs)
    x = layers.BatchNormalization(axis=bn_axis, name=name + '_conv1_bn1')(x)
    x = layers.Activation(activation='relu', name=name + '_conv1_act')(x)
    # group conv 3*3
    x = group_conv(
        x, group_channels=group_channels, cardinality=cardinality, strides=1, weight_decay=weight_decay, name=name)
    # conv 1*1
    x = layers.Conv2D(2 * filters, kernel_size=1, padding='same', use_bias=False, name=name + '_conv4')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=name + '_conv4_bn')(x)
    # residual
    x = layers.add([init, x])
    x = layers.Activation(activation='relu', name=name + '_conv4_act')(x)

    return x


# create ResNetXt
# 为了简洁， 网络没有加初始化和正则化，可根据卷积自行调整
def ResNetXt(input_shape, classes, cardinality=8, blocks=None):
    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1
    img_input = keras.Input(shape=input_shape, name='inputs')

    x = initial_conv_block(img_input, weight_decay=5e-4)
    # blocks1
    for i in range(blocks[0]):
        x = bottlencect_block(x, filters=2 * 64, cardinality=cardinality, strides=1, weight_decay=5e-4,
                              name='blocks_1_%d' % i)
    # blocks数 2~4
    for i, b in enumerate(blocks[1:]):
        f = 2 ** (i + 2) # 控制filters
        i = i + 2 # 控制blocks id
        for j in range(b):
            # block的第一层，图片减小一倍---同resnet
            if j == 0:
                x = bottlencect_block(x, filters=64 * f, cardinality=cardinality, strides=2, weight_decay=5e-4,
                                      name='blocks_%d_%d' % (i, j))
            else:
                x = bottlencect_block(x, filters=64 * f, cardinality=cardinality, strides=1, weight_decay=5e-4,
                                      name='blocks_%d_%d' % (i, j))

    x = layers.GlobalAveragePooling2D(name='features')(x)
    x = layers.Dense(classes, use_bias=False, activation='softmax', name='classes')(x)

    model = training.Model(img_input, x, name='ResNetXt')

    return model


if __name__ == '__main__':
    x_train, y_train, x_test, y_test = read_data()
    # 每一个blocks的重复次数 50:[3, 4, 6, 3]  101: [3, 4, 23, 3]
    blocks = [3, 4, 23, 3]
    model = ResNetXt(input_shape=input_shape, classes=num_classes, blocks=blocks)
    # 编译模型
    opt = keras.optimizers.Adam(learning_rate=1e-3, epsilon=1e-7)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x=x_train, y=y_train, batch_size=500, epochs=10, validation_split=0.2, validation_data=[x_test, y_test])
    model.summary()

    pre = model.evaluate(x_test, y_test, batch_size=500)
    print('test_loss: ', pre[0], 'test_acc: ', pre[1])
