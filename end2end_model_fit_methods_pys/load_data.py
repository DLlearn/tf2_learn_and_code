#! /usr/bin/env python
# -*- coding:utf-8 -*-
import os
import tensorflow as tf
import numpy as np

def load1(path, batch_size = 64, subset = None, train = True):
    data_pattern = os.path.join(path, subset)
    data_files = tf.io.gfile.glob(data_pattern)

    if len(data_files) < 1:
        raise Exception('[ERROR] No train data files found in %s' % path)

    total_samples = sum([os.path.getsize(f) / 4 / 2040 for f in data_files])
    dataset = tf.data.FixedLengthRecordDataset(filenames = data_files,
                                               record_bytes = 4 * 2040)
    if train:
        dataset.shuffle(buffer_size =10000, reshuffle_each_iteration = True)

    def transfer(value):
        value = tf.io.decode_raw(value, tf.float32)
        label = tf.cast(tf.slice(value, [0], [40]), tf.int32)
        data = tf.slice(value, [40], [2000])
        data = tf.reshape(data, [2000, 1])
        return data, label

    dataset = dataset.map(transfer,num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if train:
        batched_dataset = dataset.batch(batch_size).prefetch(buffer_size = tf.data.experimental.AUTOTUNE).repeat()
    else:
        batched_dataset = dataset.batch(batch_size).prefetch(buffer_size = tf.data.experimental.AUTOTUNE).repeat(1)

    return batched_dataset, total_samples




def load2(path, batch_size = 64, subset = None, train = True):
    data_pattern = os.path.join(path, subset)
    data_files = tf.io.gfile.glob(data_pattern)

    if len(data_files) < 1:
        raise Exception('[ERROR] No train data files found in %s' % path)

    total_samples = sum([os.path.getsize(f) / 4 / 2040 for f in data_files])
    dataset = tf.data.FixedLengthRecordDataset(filenames = data_files,
                                               record_bytes = 4 * 2040)
    if train:
        dataset.shuffle(buffer_size =10000, reshuffle_each_iteration = True)

    def transfer(value):
        value = tf.io.decode_raw(value, tf.float32)
        label = tf.cast(tf.slice(value, [0], [40]), tf.int32)
        data = tf.slice(value, [40], [2000])
        data = tf.reshape(data, [2000, 1])
        return data, label

    dataset = dataset.map(transfer,num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if train:
        batched_dataset = dataset.batch(batch_size).prefetch(buffer_size = tf.data.experimental.AUTOTUNE).repeat()
    else:
        batched_dataset = dataset.batch(batch_size).prefetch(buffer_size = tf.data.experimental.AUTOTUNE).repeat(1)

    return batched_dataset, total_samples
def load3(path, batch_size = 64,leader="II" ,subset = None, train = True):
    data_pattern = os.path.join(path, subset)
    data_files = tf.io.gfile.glob(data_pattern)
    print("data files:",data_files)
    if len(data_files) < 1:
        raise Exception('[ERROR] No train data files found in %s' % path)

    total_samples = sum([os.path.getsize(f) / 4 / 8121 for f in data_files])
    dataset = tf.data.FixedLengthRecordDataset(filenames = data_files,
                                               record_bytes = 4 * 8121)
    if train:
        dataset.shuffle(buffer_size =10000, reshuffle_each_iteration = True)

    def transfer(value):
        value = tf.io.decode_raw(value, tf.float32)
        label = tf.cast(tf.slice(value, [1], [40]), tf.int32)

        if leader.lower()=='i':
            data = tf.reshape(tf.slice(value, [121], [2000]), [2000, 1])
        elif leader.lower()=='ii':
            data = tf.reshape(tf.slice(value, [2121], [2000]), [2000, 1])
        elif leader.lower()=='v1':
            data = tf.reshape(tf.slice(value, [4121], [2000]), [2000, 1])
        elif leader.lower()=='v5':
            data = tf.reshape(tf.slice(value, [6121], [2000]), [2000, 1])
        elif leader.lower()=='all':
            data = tf.reshape(tf.slice(value, [121], [8000]), [8000, 1])
        else:
            raise Exception('Leader is wrong')
        return data, label

    dataset = dataset.map(transfer,num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if train:
        batched_dataset = dataset.batch(batch_size).prefetch(buffer_size = tf.data.experimental.AUTOTUNE).repeat()
    else:
        batched_dataset = dataset.batch(batch_size).prefetch(buffer_size = tf.data.experimental.AUTOTUNE).repeat(1)

    return batched_dataset, total_samples

def load4(path, batch_size = 64,leader="II" ,subset = None, train = True):
    data_pattern = os.path.join(path, subset)
    data_files = tf.io.gfile.glob(data_pattern)
    print("data files:",data_files)
    if len(data_files) < 1:
        raise Exception('[ERROR] No train data files found in %s' % path)

    total_samples = sum([os.path.getsize(f) / 4 / 8121 for f in data_files])
    dataset = tf.data.FixedLengthRecordDataset(filenames = data_files,
                                               record_bytes = 4 * 8121)
    if train:
        dataset.shuffle(buffer_size =10000, reshuffle_each_iteration = True)

    def transfer(value):
        max_len=40 #最大心跳个数
        value = tf.io.decode_raw(value, tf.float32)
        label = tf.cast(tf.slice(value, [1], [40]), tf.int32)
        mask = tf.cast(tf.slice(value, [81], [40]), tf.int32)
        heart_label= tf.reshape(tf.gather(label,tf.where(tf.math.greater(mask,tf.constant(0)))),[-1])
        # heart_label = (label+1)*mask
        # heart_label = tf.reshape(tf.boolean_mask(heart_label,tf.greater(heart_label,tf.constant(0,dtype=tf.int32))),(1,-1))-1
        # heart_label = tf.concat(([[heart_label.shape[1]]], heart_label),axis=-1)
        # paddings = [[0,0],[0,max_len-heart_label.shape[1]]]
        # heart_label=tf.pad(heart_label,paddings,'CONSTNAT',constant_values=-1)


        if leader.lower()=='i':
            data = tf.reshape(tf.slice(value, [121], [2000]), [2000, 1])
        elif leader.lower()=='ii':
            data = tf.reshape(tf.slice(value, [2121], [2000]), [2000, 1])
        elif leader.lower()=='v1':
            data = tf.reshape(tf.slice(value, [4121], [2000]), [2000, 1])
        elif leader.lower()=='v5':
            data = tf.reshape(tf.slice(value, [6121], [2000]), [2000, 1])
        elif leader.lower()=='all':
            data = tf.reshape(tf.slice(value, [121], [8000]), [8000, 1])
        else:
            raise Exception('Leader is wrong')
        return data, label,mask,heart_label

    dataset = dataset.map(transfer,num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if leader.lower() in ['i','ii','v1','v5']:
        padded_shapes=([2000,1],[40],[40],[1,20])
    elif leader.lower()=='all':
        padded_shapes = ([8000, 1], [40], [40], [20])
    padding_values=((-1.0,-1,-1,-1))
    dataset = dataset.padded_batch(batch_size,padded_shapes = padded_shapes,padding_values = padding_values)
    if train:
        batched_dataset = dataset.prefetch(buffer_size = tf.data.experimental.AUTOTUNE).repeat()
    else:
        batched_dataset = dataset.prefetch(buffer_size = tf.data.experimental.AUTOTUNE).repeat(1)

    return batched_dataset, total_samples


#train_ds, total_samples = load(train_data_path,batch_size=32,subset='train_*')
if __name__=="__main__":
    train_data_path = "/deeplearn_data2/experimental_data/191108/origin_datas"
    batch_size =32
    train_ds, total_samples = load1(train_data_path, batch_size = batch_size, subset = 'train_*')
    for data,label in train_ds.take(1):
        d = data.numpy()
        l = label.numpy()
        print("data.shape",d.shape)
        print("label.shape",l.shape)
        print(np.max(l))
        print(np.min(l))
        print(np.max(d))
        print(np.min(d))
