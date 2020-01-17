#! /usr/bin/env python
# -*- coding:utf-8 -*-
import tensorflow as tf
def compute_loss(labels,logits,num_classes):
    labels=tf.one_hot(labels,depth=num_classes)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True,label_smoothing=0.1)(labels,logits,sample_weight=None)
    return loss