#! /usr/bin/env python
# -*- coding:utf-8 -*-
#! /usr/bin/env python
# -*- coding:utf-8 -*-
import tensorflow as tf
from functools import partial

l2 = tf.keras.regularizers.l2(1e-3)#定义模型正则化方法
conv2d = partial(tf.keras.layers.Conv2D,activation='relu',padding='valid',kernel_regularizer=l2,bias_regularizer=l2)
fc = partial(tf.keras.layers.Dense,activation='relu',kernel_regularizer=l2,bias_regularizer=l2)
maxpool=tf.keras.layers.MaxPooling2D
dropout=tf.keras.layers.Dropout


def block(inputs,fir,sec,thir,four,name):
    with tf.name_scope(name):
        conv5=conv2d(fir,[1,5],1,padding='same')(inputs)
        conv10=conv2d(sec,[1,10],1,padding='same')(inputs)
        conv20=conv2d(thir,[1,20],1,padding='same')(inputs)
        pool1 = maxpool([1,5],1,padding='same')(inputs)
        pool2 = conv2d(four,[1,1],1,padding='same')(pool1)
        out = tf.keras.layers.concatenate([conv5,conv10,conv20,pool2],axis=-1)
        return out
def infer(num_classes):
    input_node = tf.keras.layers.Input(shape=(1,1000,1),name='input_node')
    x = conv2d(64,[1,5],[1,1],name='conv1')(input_node)
    x = maxpool([1,3],[1,2],name='pool1')(x)
    x = conv2d(96,[1,5],[1,1],name='conv2')(x)
    x = maxpool([1,3],[1,2],name='pool2')(x)
    x = block(x, 32, 64, 16, 16, 'mixed_128')
    x = block(x, 64, 96, 48, 32, 'mixed_240')
    x = maxpool([1, 3], [1, 2], name = 'pool_after_mixed_240')(x)
    x = block(x, 96, 112, 24, 32, 'mixed_264')
    x = block(x, 80, 128, 32, 32, 'mixed_272_1')
    x = maxpool([1, 3], [1, 2], name = 'pool_after_mixed_272_1')(x)
    x = block(x, 64, 144, 32, 32, 'mixed_272_2')
    x = block(x, 96, 160, 64, 64, 'mixed_384_1')
    x = maxpool([1, 3], [1, 2], name = 'pool_after_mixed_384_1')(x)
    x = block(x, 96, 160, 64, 64, 'mixed_384_2')
    x = block(x, 128, 192, 80, 80, 'mixed_480')
    x = maxpool([1, 3], [1, 2], name = 'pool_after_mixed_480')(x)
    x = conv2d(256,[1,5],[1,1],name='conv_end')(x)
    x = tf.keras.layers.Flatten()(x)
    x = fc(512,name='fc1')(x)
    x = fc(512,name='fc2')(x)
    x = dropout(0.5)(x)
    x = fc(num_classes,activation=None,name='logits')(x)

    model = tf.keras.models.Model(inputs=[input_node],outputs=[x])
    return model