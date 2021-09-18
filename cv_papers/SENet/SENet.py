# -*- coding: utf-8 -*-
# @File : SE_ResNet.py
# @Author: snailfrying
# @Time : 2021/9/12 10:30
# @Software: PyCharm

import numpy as np

from tensorflow import keras
from tensorflow.keras import layers, utils
from tensorflow.python.keras import backend
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import ResNet50, ResNet101, ResNet152
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

def ResNet_50_ym(input_shape, classes):
    # 您可以从keras查看大于50layers的ResNet结构的看源码！！！
    # 由于keras没有实现18 34的我这实现了，并且其他模型也基本展示了结构
    # 主要查看block的不一样，并且我会结合SENet从新实现ResNet,可以运行查看不同。
    model = ResNet50(input_shape=input_shape, classes=classes, weights=None)
    return model

# 定义 se-resnet 18 34 layers的结构
def se_identity_block(x, filters, kernel_size=3, stride=1, conv_shortcut=False, reduce_ratio=8, name=None):
    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1
    if conv_shortcut:
        shortcut = layers.Conv2D(filters, kernel_size=1, strides=stride, padding='same', name=name + '_0_conv')(x)
        shortcut = layers.BatchNormalization(
            axis=bn_axis, epsilon=1.001e-5, name=name + '_0_bn')(shortcut)
    else:
        shortcut = x

    x = layers.Conv2D(filters, kernel_size, strides=stride, padding='same', name=name + '_1_conv')(x)
    x = layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name=name + '_1_bn')(x)
    x = layers.Activation('relu', name=name + '_1_relu')(x)

    # SE
    se_input = x
    x = layers.GlobalAveragePooling2D()(se_input)
    x = layers.Reshape((1, 1, filters))(x)
    x = layers.Dense(filters // reduce_ratio, activation='relu')(x)
    x = layers.Dense(filters, activation='sigmoid')(x)
    x = layers.Multiply()([x, se_input])

    x = layers.Add(name=name + '_add')([x, shortcut])
    x = layers.Activation('relu', name=name + '_out')(x)

    return x

# 定义 se-resnet 50 101 152 layers的结构
def se_conv_block(x, filters, kernel_size=3, stride=1, conv_shortcut=True, reduce_ratio=8, name=None):
    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

    if conv_shortcut:
        shortcut = layers.Conv2D(filters, kernel_size=1, strides=stride, padding='same', name=name + '_0_conv')(x)
        shortcut = layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name=name + '_0_bn')(shortcut)
    else:
        shortcut = x

    x = layers.Conv2D(
      filters, 1, strides=stride, name=name + '_1_conv')(x)
    x = layers.BatchNormalization(
      axis=bn_axis, epsilon=1.001e-5, name=name + '_1_bn')(x)
    x = layers.Activation('relu', name=name + '_1_relu')(x)

    x = layers.Conv2D(
      filters, kernel_size, padding='SAME', name=name + '_2_conv')(x)
    x = layers.BatchNormalization(
      axis=bn_axis, epsilon=1.001e-5, name=name + '_2_bn')(x)
    x = layers.Activation('relu', name=name + '_2_relu')(x)

    x = layers.Conv2D(filters, 1, name=name + '_3_conv')(x)
    x = layers.BatchNormalization(
      axis=bn_axis, epsilon=1.001e-5, name=name + '_3_bn')(x)

    # SE
    se_input = x
    x = layers.GlobalAveragePooling2D()(se_input)
    x = layers.Reshape((1, 1, filters))(x)
    x = layers.Dense(filters // reduce_ratio, activation='relu')(x)
    x = layers.Dense(filters, activation='sigmoid')(x)
    x = layers.Multiply()([x, se_input])

    x = layers.Add(name=name + '_add')([shortcut, x])
    x = layers.Activation('relu', name=name + '_out')(x)
    return x

def identity_stack(x, filters, blocks, stride1=2, conv_shortcut=False, name=None):
    x = se_identity_block(x, filters, stride=stride1, conv_shortcut=conv_shortcut, name=name)
    for i in range(0, blocks):
        x = se_identity_block(x, filters, stride=1, name=name + '_block' + str(i))
    return x

def conv_stack(x, filters, blocks, stride1=2, conv_shortcut=False, name=None):
    x = se_conv_block(x, filters, stride=stride1, conv_shortcut=conv_shortcut, name=name)
    for i in range(1, blocks):
        x = se_conv_block(x, filters, stride=1, name=name + '_block' + str(i))
    return x

def ResNet_18(input_shape, classes):

    image_inputs = keras.Input(input_shape)
    x = layers.ZeroPadding2D(padding=(3, 3))(image_inputs)
    # block 1
    x = layers.Conv2D(filters=64, kernel_size=7, strides=2, padding='valid')(x)
    x = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(x)
    # block 2  ResNet-34: blocks=2
    x = identity_stack(x, filters=128, blocks=2, stride1=2, conv_shortcut=True, name='block2')
    # block 3  ResNet-34: blocks=3
    x = identity_stack(x, filters=256, blocks=2, stride1=2, conv_shortcut=True, name='block3')
    # block 4  ResNet-34: blocks=5
    x = identity_stack(x, filters=512, blocks=2, stride1=2, conv_shortcut=True, name='block4')
    # block 5  ResNet-34: blocks=2
    x = layers.AveragePooling2D(pool_size=(7, 7))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(units=classes, activation='softmax')(x)

    model = training.Model(image_inputs, x, name='se-resnet-18')
    return model

def ResNet_50(input_shape, classes):
    image_inputs = keras.Input(input_shape)
    image_inputs = keras.Input(input_shape)
    x = layers.ZeroPadding2D(padding=(3, 3))(image_inputs)
    # block 1
    x = layers.Conv2D(filters=64, kernel_size=7, strides=2, padding='valid')(x)
    x = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

    x = conv_stack(x, filters=64,  blocks=3, stride1=1, conv_shortcut=True, name='conv2')
    x = conv_stack(x, filters=128, blocks=4, stride1=1, conv_shortcut=True, name='conv3')
    x = conv_stack(x, filters=256, blocks=6, stride1=1, conv_shortcut=True, name='conv4')
    x = conv_stack(x, filters=512, blocks=3, stride1=1, conv_shortcut=True, name='conv5')

    x = layers.GlobalAvgPool2D(name='avg_pool')(x)
    x = layers.Dense(num_classes, name="prediction", activation="softmax")(x)

    model = training.Model(image_inputs, x, name='se-resnet-50')
    return model

def ResNet_101(input_shape, classes):
    image_inputs = keras.Input(input_shape)
    image_inputs = keras.Input(input_shape)
    x = layers.ZeroPadding2D(padding=(3, 3))(image_inputs)
    # block 1
    x = layers.Conv2D(filters=64, kernel_size=7, strides=2, padding='valid')(x)
    x = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

    x = conv_stack(x, filters=64,  blocks=3, stride1=1, conv_shortcut=True, name='conv2')
    x = conv_stack(x, filters=128, blocks=4, stride1=1, conv_shortcut=True, name='conv3')
    x = conv_stack(x, filters=256, blocks=23, stride1=1, conv_shortcut=True, name='conv4')
    x = conv_stack(x, filters=512, blocks=3, stride1=1, conv_shortcut=True, name='conv5')

    x = layers.GlobalAvgPool2D(name='avg_pool')(x)
    x = layers.Dense(num_classes, name="prediction", activation="softmax")(x)

    model = training.Model(image_inputs, x, name='se-resnet-101')
    return model
def ResNet_152(input_shape, classes):
    image_inputs = keras.Input(input_shape)
    image_inputs = keras.Input(input_shape)
    x = layers.ZeroPadding2D(padding=(3, 3))(image_inputs)
    # block 1
    x = layers.Conv2D(filters=64, kernel_size=7, strides=2, padding='valid')(x)
    x = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

    x = conv_stack(x, filters=64,  blocks=3, stride1=1, conv_shortcut=True, name='conv2')
    x = conv_stack(x, filters=128, blocks=8, stride1=1, conv_shortcut=True, name='conv3')
    x = conv_stack(x, filters=256, blocks=36, stride1=1, conv_shortcut=True, name='conv4')
    x = conv_stack(x, filters=512, blocks=3, stride1=1, conv_shortcut=True, name='conv5')

    x = layers.GlobalAvgPool2D(name='avg_pool')(x)
    x = layers.Dense(num_classes, name="prediction", activation="softmax")(x)

    model = training.Model(image_inputs, x, name='se-resnet-152')
    return model

if __name__ == '__main__':
    x_train, y_train, x_test, y_test = read_data()

    model = ResNet_50(input_shape=input_shape, classes=num_classes)
    # 编译模型
    opt = keras.optimizers.Adam(learning_rate=1e-3, epsilon=1e-7)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    model.fit(x=x_train, y=y_train, batch_size=10, epochs=2, validation_split=0.2, validation_data=[x_test, y_test])
    model.summary()

    pre = model.evaluate(x_test, y_test, batch_size=500)
    print('test_loss: ', pre[0], 'test_acc: ', pre[1])
