#! /usr/bin/env python
# -*- coding:utf-8 -*-
import tensorflow as tf
from functools import partial

l2 = tf.keras.regularizers.l2(1e-3)#定义模型正则化方法
conv1d = partial(tf.keras.layers.Conv1D,activation='relu',padding='same',kernel_regularizer=l2,bias_regularizer=l2)
fc = partial(tf.keras.layers.Dense,activation='relu',kernel_regularizer=l2,bias_regularizer=l2)
maxpool=tf.keras.layers.MaxPooling1D
dropout=tf.keras.layers.Dropout
lstm = partial(tf.keras.layers.LSTM,kernel_regularizer=l2,bias_regularizer=l2)

def model(num_classes):
    input_node = tf.keras.layers.Input(shape=(2000,1))
    x = conv1d(64,10)(input_node)
    x = maxpool(5,2,padding='same')(x)
    x = conv1d(128,10)(x)
    x = maxpool(5,2,padding='same')(x)
    x = conv1d(128,5)(x)
    x = maxpool(5,2,padding='same')(x)
    x = conv1d(256,5)(x)
    x = maxpool(5,2,padding='same')(x)
    x = conv1d(256,5)(x)
    x = maxpool(5,2,padding='same')(x)
    x = conv1d(256,5)(x)
    x = maxpool(5,2,padding='same')(x)
    x = lstm(64)(x)
    x = tf.keras.layers.RepeatVector(40)(x)
    x = lstm(128,return_sequences=True)(x)
    x = fc(128)(x)
    x = dropout(0.5)(x)
    x = fc(num_classes,activation=None)(x)
    model = tf.keras.models.Model(inputs=[input_node],outputs=[x])
    return model