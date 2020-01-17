#! /usr/bin/env python
# -*- coding:utf-8 -*-
import tensorflow as tf
from functools import partial

l2 = tf.keras.regularizers.l2(1e-3)#定义模型正则化方法
conv2d = partial(tf.keras.layers.Conv2D,activation='relu',padding='valid',kernel_regularizer=l2,bias_regularizer=l2)
fc = partial(tf.keras.layers.Dense,activation='relu',kernel_regularizer=l2,bias_regularizer=l2)
maxpool=tf.keras.layers.MaxPooling2D
dropout=tf.keras.layers.Dropout
lstm = partial(tf.keras.layers.LSTM,kernel_regularizer=l2,bias_regularizer=l2)
def block(inputs,fir,sec,thir,four,name):
    with tf.name_scope(name):
        conv5=conv2d(fir,[1,5],1,padding='same')(inputs)
        conv5 = tf.keras.layers.BatchNormalization()(conv5)
        conv10=conv2d(sec,[1,9],1,padding='same')(inputs)
        conv10 = tf.keras.layers.BatchNormalization()(conv10)
        conv20=conv2d(thir,[1,15],1,padding='same')(inputs)
        conv20 = tf.keras.layers.BatchNormalization()(conv20)
        pool1 = maxpool([1,5],1,padding='same')(inputs)
        pool2 = conv2d(four,[1,1],1,padding='same')(pool1)
        pool2 = tf.keras.layers.BatchNormalization()(pool2)
        out = tf.keras.layers.concatenate([conv5,conv10,conv20,pool2],axis=-1)
        return out
def infer(num_classes,name):
    input_node = tf.keras.layers.Input(shape = (1, 2000, 1), name = 'inputs')
    x = conv2d(128, [1, 10])(input_node)
    x = maxpool([1, 3], [1, 2], padding = 'valid')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = conv2d(256, [1, 10])(x)
    x = maxpool([1, 3], [1, 2], padding = 'valid')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = conv2d(256, [1, 10])(x)
    x = maxpool([1, 3], [1, 2], padding = 'valid')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = conv2d(256, [1, 10])(x)
    x = maxpool([1, 3], [1, 2], padding = 'valid')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = conv2d(256, [1, 10])(x)
    x = maxpool([1, 3], [1, 2], padding = 'valid')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = conv2d(256, [1, 10])(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = conv2d(256, [1, 4])(x)
    x = tf.keras.layers.BatchNormalization()(x)
    shape = x.shape
    print(shape)
    x = tf.keras.layers.Reshape([shape[2], shape[3]])(x)
    x = lstm(256, return_sequences = True)(x)
    x = lstm(256, return_sequences = True)(x)
    x = fc(256)(x)
    x = dropout(0.2)(x)
    x = fc(128)(x)
    x = dropout(0.2)(x)
    x = fc(num_classes, activation = None, name = 'outputs')(x)

    model = tf.keras.models.Model(input_node, x,name=name)
    return model