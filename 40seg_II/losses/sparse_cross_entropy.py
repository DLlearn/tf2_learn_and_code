#! /usr/bin/env python
# -*- coding:utf-8 -*-
import tensorflow as tf
def compute_loss(labels,logits):
    # loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels,logits)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)(labels,logits,sample_weight=None)
    return loss