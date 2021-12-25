# -*- coding: utf-8 -*-
# @File : DenseNet.py
# @Author: snailfrying
# @Time : 2021/9/12 10:30
# @Software: PyCharm

import numpy as np

from tensorflow import keras
from tensorflow.keras import layers, utils
from tensorflow.python.keras import backend
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import DenseNet121, DenseNet201
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

def Dense_201(input_shape, classes):
    # 结构直接看源码！！！
    model = DenseNet201(input_shape=input_shape, classes=classes, weights=None)
    return model

# 这其实就是kera实现Dense系列的源码。
# 只是用闭包的形式展出，方便自己调试
# 建议把conv block中relu激活去掉，看对模型的影响。
def Dense_121(input_shape, classes):
    def dense_block(x, blocks, name):

        for i in range(blocks):
            x = conv_block(x, 32, name=name + '_block' + str(i + 1))
        return x

    def transition_block(x, reduction, name):

        bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1
        x = layers.BatchNormalization(
            axis=bn_axis, epsilon=1.001e-5, name=name + '_bn')(
            x)
        x = layers.Activation('relu', name=name + '_relu')(x)
        x = layers.Conv2D(
            int(backend.int_shape(x)[bn_axis] * reduction),
            1,
            use_bias=False,
            name=name + '_conv')(
            x)
        x = layers.AveragePooling2D(2, strides=2, name=name + '_pool')(x)
        return x

    def conv_block(x, growth_rate, name):

        bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1
        x1 = layers.BatchNormalization(
            axis=bn_axis, epsilon=1.001e-5, name=name + '_0_bn')(
            x)
        x1 = layers.Activation('relu', name=name + '_0_relu')(x1)
        x1 = layers.Conv2D(
            4 * growth_rate, 1, use_bias=False, name=name + '_1_conv')(
            x1)
        x1 = layers.BatchNormalization(
            axis=bn_axis, epsilon=1.001e-5, name=name + '_1_bn')(
            x1)
        x1 = layers.Activation('relu', name=name + '_1_relu')(x1)
        x1 = layers.Conv2D(
            growth_rate, 3, padding='same', use_bias=False, name=name + '_2_conv')(
            x1)
        x = layers.Concatenate(axis=bn_axis, name=name + '_concat')([x, x1])
        return x

    # 可直接查看论文，验证模型架构与实现的区别。
    # 121: [6, 12, 24, 16], 169: [6, 12, 32, 32], 201: [6, 12, 48, 32]
    blocks = [6, 12, 24, 16]
    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

    image_inputs = keras.Input(input_shape)
    x = layers.ZeroPadding2D(padding=((3, 3), (3, 3)))(image_inputs)
    x = layers.Conv2D(64, 7, strides=2, use_bias=False, name='conv1/conv')(x)
    x = layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name='conv1/bn')(
        x)
    x = layers.Activation('relu', name='conv1/relu')(x)
    x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
    x = layers.MaxPooling2D(3, strides=2, name='pool1')(x)

    x = dense_block(x, blocks[0], name='conv2')
    x = transition_block(x, 0.5, name='pool2')
    x = dense_block(x, blocks[1], name='conv3')
    x = transition_block(x, 0.5, name='pool3')
    x = dense_block(x, blocks[2], name='conv4')
    x = transition_block(x, 0.5, name='pool4')
    x = dense_block(x, blocks[3], name='conv5')

    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name='bn')(x)
    x = layers.Activation('relu', name='relu')(x)

    x = layers.GlobalAveragePooling2D(name='avg_pool')(x)

    x = layers.Dense(classes, activation='softmax',
                     name='predictions')(x)
    model = training.Model(image_inputs, x, name='DenseNet')
    return model


if __name__ == '__main__':
    x_train, y_train, x_test, y_test = read_data()

    model = Dense_121(input_shape=input_shape, classes=num_classes)
    # 编译模型
    opt = keras.optimizers.Adam(learning_rate=1e-3, epsilon=1e-7)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    print(x_train.shape, y_train.shape,x_test.shape, y_test.shape)
    model.fit(x=x_train, y=y_train, batch_size=10, epochs=2, validation_split=0.2, validation_data=[x_test, y_test])
    model.summary()

    pre = model.evaluate(x_test, y_test, batch_size=500)
    print('test_loss: ', pre[0], 'test_acc: ', pre[1])
