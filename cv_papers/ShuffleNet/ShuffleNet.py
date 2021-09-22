# -*- coding: utf-8 -*-
# @File : ShuffleNet.py
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

    x = layers.Conv2D(24, (3, 3), strides=2, padding='same', use_bias=False, name='stage1_conv')(inputs)
    x = layers.MaxPooling2D(pool_size=3, strides=2, padding='same', name='stage1_maxpool')(x)
    x = layers.BatchNormalization(axis=bn_axis, name='stage1_conv_bn')(x)
    x = layers.Activation('relu', name='stage1_conv_act')(x)
    return x


# 建立group convolution，即根据cardinality，把输入特征划分cardinality个group,然后对每一个group使用channels-filters提取特征，最后合并。
def group_conv(inputs, filters, kernel_size=3, strides=1, cardinality=8, padding='same', weight_decay=5e-4,
               name=None):
    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1
    in_channels = backend.int_shape(inputs)[bn_axis] // cardinality
    out_channels = filters // cardinality
    if cardinality == 1:
        x = layers.Conv2D(out_channels, kernel_size=kernel_size, strides=strides, padding=padding,
                          name=name + '_conv2')(inputs)
        x = layers.BatchNormalization(axis=bn_axis, name=name + '_conv2_bn')(x)
        x = layers.Activation(activation='relu', name=name + '_conv2_act')(x)
        return x

    feature_map_list = []
    for c in range(cardinality):
        x = layers.Lambda(
            lambda z: z[:, :, :, c * in_channels:(c + 1) * in_channels]
            if backend.image_data_format() == 'channels_last'
            else lambda z: z[:, c * in_channels:(c + 1) * in_channels, :, :]
        )(inputs)

        x = layers.Conv2D(filters=out_channels, kernel_size=kernel_size, strides=strides, padding=padding,
                          name=name + '_groupconv3_' + str(c))(x)
        feature_map_list.append(x)
    x = layers.concatenate(feature_map_list, axis=bn_axis, name=name + '_groupconv3_concat')
    x = layers.BatchNormalization(axis=bn_axis, name=name + '_groupconv3_bn')(x)
    x = layers.Activation(activation='relu', name=name + '_groupconv3_act')(x)
    return x


# 打乱通道，让各个通道有相关性。
def channel_shuffle(inputs, cardinality):
    if backend.image_data_format() == 'channels_last':
        height, width, in_channels = backend.int_shape(inputs)[1:]
        channels_per_group = in_channels // cardinality
        pre_shape = [-1, height, width, cardinality, channels_per_group]
        dim = (0, 1, 2, 4, 3)
        later_shape = [-1, height, width, in_channels]
    else:
        in_channels, height, width = backend.int_shape(inputs)[1:]
        channels_per_group = in_channels // cardinality
        pre_shape = [-1, cardinality, channels_per_group, height, width]
        dim = (0, 2, 1, 3, 4)
        later_shape = [-1, in_channels, height, width]

    x = layers.Lambda(lambda z: backend.reshape(z, pre_shape))(inputs)
    x = layers.Lambda(lambda z: backend.permute_dimensions(z, dim))(x)
    x = layers.Lambda(lambda z: backend.reshape(z, later_shape))(x)

    return x


# 构建ShuffleNet里每一个blocks,
def ShuffleNet_unit(
        inputs, filters=64, kernel_size=1, stage=1, bottleneck_ratio=1, cardinality=8, strides=1, weight_decay=5e-4,
        name=None):
    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1
    bottleneck_channels = int(filters * bottleneck_ratio)
    in_channels = backend.int_shape(inputs)[bn_axis]
    if stage == 2:
        x = layers.Conv2D(
            filters=bottleneck_channels, kernel_size=kernel_size, strides=1, padding='same', use_bias=False)(inputs)
    else:
        x = group_conv(inputs, filters=bottleneck_channels, kernel_size=1, strides=1, cardinality=cardinality,
                       name=name + '_group_conv1')

    x = layers.BatchNormalization(axis=bn_axis)(x)
    x = layers.Activation(activation='relu')(x)

    x = channel_shuffle(x, cardinality)

    x = layers.DepthwiseConv2D(kernel_size=kernel_size, strides=strides, depth_multiplier=1,
                               padding='same', use_bias=False)(x)

    if strides == 2:
        x = group_conv(x, filters - in_channels, kernel_size=1, strides=1, cardinality=cardinality,
                       name=name + '_group_conv2')
        x = layers.BatchNormalization(axis=bn_axis)(x)
        avg = layers.AveragePooling2D(pool_size=3, strides=2, padding='same')(inputs)
        x = layers.Concatenate(axis=bn_axis)([x, avg])
    else:
        x = group_conv(x, filters, kernel_size=1, strides=1, cardinality=cardinality, name=name + '_group_conv3')
        x = layers.BatchNormalization(axis=bn_axis)(x)
        x = layers.add([x, inputs])
    return x


# 训练各个stage
def shuffle_stage(inputs, filters, kernel_size, cardinality, repeat, stage, name=None):
    x = ShuffleNet_unit(inputs, filters, kernel_size, strides=2, cardinality=cardinality, stage=stage, name=name)
    for i in range(1, repeat):
        x = ShuffleNet_unit(x, filters, kernel_size, strides=1, cardinality=cardinality, stage=stage,
                            name=name + '_' + str(i))
    return x


# create ShuffleNet
# 为了简洁， 网络没有加初始化和正则化，可根据卷积自行调整
def ShuffleNet(input_shape, classes, cardinality=8, blocks=None):
    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1
    img_input = keras.Input(shape=input_shape, name='inputs')

    # stage 1
    x = initial_conv_block(img_input, weight_decay=5e-4)
    # repeats 每一个blocks的重复次数
    repeats = [4, 8, 4]
    for b in range(cardinality):
        print('stage %d' % b)
        x = shuffle_stage(x, filters=blocks[b], kernel_size=3, cardinality=cardinality, repeat=repeats[b], stage=b + 2,
                          name='stage_' + str(b + 2))

    x = layers.GlobalAveragePooling2D(name='features')(x)
    x = layers.Dense(classes, use_bias=False, activation='softmax', name='classes')(x)

    model = training.Model(img_input, x, name='ShuffleNet')

    return model


if __name__ == '__main__':
    x_train, y_train, x_test, y_test = read_data()
    '''
        cardinality=groups: 1, [144, 288, 576]
        cardinality=groups: 2, [200, 400, 800]
        cardinality=groups: 3, [240, 480, 960]
        cardinality=groups: 4, [272, 544, 1088]
        cardinality=groups: 8, [384, 768, 1536]
    '''
    cardinality, blocks = 1, [144, 288, 576]
    model = ShuffleNet(input_shape=input_shape, classes=num_classes, blocks=blocks, cardinality=cardinality)
    # 编译模型
    opt = keras.optimizers.Adam(learning_rate=1e-3, epsilon=1e-7)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x=x_train, y=y_train, batch_size=500, epochs=10, validation_split=0.2, validation_data=[x_test, y_test])
    model.summary()

    pre = model.evaluate(x_test, y_test, batch_size=500)
    print('test_loss: ', pre[0], 'test_acc: ', pre[1])
