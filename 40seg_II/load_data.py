#! /usr/bin/env python
# -*- coding:utf-8 -*-
import os
import tensorflow as tf
import numpy as np
from collections import namedtuple

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
def load5(path, batch_size = 64,subset = None, train = True,epoch=None):
    data_pattern = os.path.join(path, subset)
    data_files = tf.io.gfile.glob(data_pattern)
    print("data files:",data_files)
    if len(data_files) < 1:
        raise Exception('[ERROR] No train data files found in %s' % path)

    total_samples = sum([os.path.getsize(f) / 4 / 1001 for f in data_files])
    dataset = tf.data.FixedLengthRecordDataset(filenames = data_files,
                                               record_bytes = 4 * 1001)
    if train:
        dataset.shuffle(buffer_size =50000, reshuffle_each_iteration = True)

    def transfer(value):
        value = tf.io.decode_raw(value, tf.float32)
        label = tf.cast(tf.slice(value, [0], [1]), tf.int32)
        data = tf.reshape(tf.slice(value,[1],[1000]),(1,1000,1))


        return data, label

    dataset = dataset.map(transfer,num_parallel_calls=tf.data.experimental.AUTOTUNE)

    dataset = dataset.batch(batch_size)
    if train:
        batched_dataset = dataset.prefetch(buffer_size = tf.data.experimental.AUTOTUNE)
    else:
        batched_dataset = dataset.prefetch(buffer_size = tf.data.experimental.AUTOTUNE)

    return batched_dataset, total_samples
def load_II_old(path, batch_size,train=False):
    """
    :param path: 数据的路径
    :param batch_size: 批量大小
    :param train:是否是训练
    :return: tf.dataset
    """



    DS = namedtuple('DS', ['id','features', 'labels', 'hb_labels','hb_nums','rmask'])
    # 以下是不全是N的数据
    # path = '/deeplearn_data2/experimental_data/191210/Channel_II/NotContainNoise/NonPureN/Train'  # 全部是8个文件
    # 基础数据，片段ID+8sII ，总计2001个float32
    data_path = os.path.join(path, 'Raw/train*')
    # 心搏数据，片段ID+心搏数量+心搏类型（12分类）+padding(-1),最长40个数，总计42个数
    heart_beat_path = os.path.join(path, 'Attr/hb_labels*')
    # 干扰数据，片段ID+II导干扰掩码1正常0干扰，所以总点数是1+2000=2001
    gr_info_path = os.path.join(path, 'Attr/hb_noise*')
    # R点位置，片段ID+40个小格[-1,-1,...,位置,...]这种的非零数，总数41
    rpos_info_path = os.path.join(path, 'Attr/Segment40/seq40_RR*')
    # R点掩码，片段ID+40格，有R点的是1，无R点的是0，总数41
    rmask_info_path = os.path.join(path, 'Attr/Segment40/seq40_mask*')
    # 40个小格分类，片段ID+40个小格各个小格的,分类，总计41
    rclass_info_path = os.path.join(path, 'Attr/Segment40/seq40_labels*')
    # 获取文件
    data_files = sorted(tf.io.gfile.glob(data_path))
    heart_beat_files = sorted(tf.io.gfile.glob(heart_beat_path))
    gr_info_files = sorted(tf.io.gfile.glob(gr_info_path))
    rpos_info_files = sorted(tf.io.gfile.glob(rpos_info_path))
    rmask_info_files = sorted(tf.io.gfile.glob(rmask_info_path))
    rclass_info_files = sorted(tf.io.gfile.glob(rclass_info_path))

    def print_info(data_files, size):
        print("[INFO] files:", data_files)
        if len(data_files) < 1:
            print('[ERROR] No train data files found in %s' % path)
            exit(-1)
        total_samples = sum([os.path.getsize(f) / 4 / size for f in data_files])
        return total_samples

    total_data_size = print_info(data_files, 2001)
    total_heart_beat_size = print_info(heart_beat_files, 42)
    total_gr_size = print_info(gr_info_files, 2001)
    total_rpos_size = print_info(rpos_info_files, 41)
    total_rmask_size = print_info(rmask_info_files, 41)
    total_rclass_size = print_info(rclass_info_files, 41)
    assert total_data_size == total_heart_beat_size == total_gr_size == total_rpos_size == total_rmask_size == total_rclass_size
    # 生成dataset
    data_dataset = tf.data.FixedLengthRecordDataset(filenames = data_files, record_bytes = 4 * 2001)
    heart_beat_dataset = tf.data.FixedLengthRecordDataset(filenames = heart_beat_files, record_bytes = 4 * 42)
    gr_info_dataset = tf.data.FixedLengthRecordDataset(filenames = gr_info_files, record_bytes = 4 * 2001)
    rpos_info_dataset = tf.data.FixedLengthRecordDataset(filenames = rpos_info_files, record_bytes = 4 * 41)
    rmask_info_dataset = tf.data.FixedLengthRecordDataset(filenames = rmask_info_files, record_bytes = 4 * 41)
    rclass_info_dataset = tf.data.FixedLengthRecordDataset(filenames = rclass_info_files, record_bytes = 4 * 41)


    def transfer_dataset(value):
        raw_data = tf.io.decode_raw(value, tf.float32)
        id = tf.cast(tf.slice(raw_data, [0], [1]), tf.int32)
        data = tf.reshape(tf.slice(raw_data, [1], [2000]), (1, 2000, 1))  # 四个导联的数据


        return id, data #shape=(1,2000,1)


    def transfer_heart_beat(value):
        raw_data = tf.io.decode_raw(value, tf.float32)
        id = tf.cast(tf.slice(raw_data, [0], [1]), tf.int32)
        heart_beat_num = tf.cast(tf.slice(raw_data, [1], [1]), tf.int32)
        heart_beat = tf.cast(tf.slice(raw_data, [2], [40]),tf.int32)  # 心搏分类

        # heart_beat = tf.reshape(tf.gather(heart_beat, tf.where(tf.math.greater(heart_beat, tf.constant(0)))), [-1])
        # heart_label = tf.reshape(tf.gather(heart_beat, tf.where(tf.math.greater(heart_beat, tf.constant(0)))), [-1])
        # negative_value=tf.reshape(tf.gather(heart_beat, tf.where(tf.math.less(heart_beat, tf.constant(0)))), [-1])
        # heart_beat = tf.concat((tf.constant([12]), heart_label, tf.constant([13]),negative_value),axis = -1)
        return id, heart_beat_num, heart_beat

    def transfer_gr_info(value):
        raw_data = tf.io.decode_raw(value, tf.float32)
        id = tf.cast(tf.slice(raw_data, [0], [1]), tf.int32)
        # 四个导联各导联干扰情况,四行分别写四个导的干扰段数以及起始点
        gr_data = tf.cast(tf.slice(raw_data, [1], [2000]), tf.int32) # 四个导联各导联干扰情况
        return id, gr_data

    def transfer_rpos_info(value):
        raw_data = tf.io.decode_raw(value, tf.float32)
        id = tf.cast(tf.slice(raw_data, [0], [1]), tf.int32)
        # 40个格，在有R点的小格写入r点位置（0-1999），其它的写-1
        rpos_data = tf.cast(tf.slice(raw_data, [1], [40]), tf.int32)
        return id, rpos_data

    def transfer_rmask_info(value):
        raw_data = tf.io.decode_raw(value, tf.float32)
        id = tf.cast(tf.slice(raw_data, [0], [1]), tf.int32)
        # 40个格，在有R点的小格写入1,其它是0
        rmask_data = tf.cast(tf.slice(raw_data, [1], [40]), tf.int32)
        return id, rmask_data

    def transfer_rclass_info(value):
        raw_data = tf.io.decode_raw(value, tf.float32)
        id = tf.cast(tf.slice(raw_data, [0], [1]), tf.int32)
        # 40个格，每个小格的分类
        rclass_data = tf.cast(tf.slice(raw_data, [1], [40]), tf.int32)
        # rclass_data=tf.concat(([12], rclass_data, [13]), axis = -1)
        return id, rclass_data

    data_dataset = data_dataset.map(transfer_dataset,num_parallel_calls = tf.data.experimental.AUTOTUNE)
    heart_beat_dataset = heart_beat_dataset.map(transfer_heart_beat,num_parallel_calls = tf.data.experimental.AUTOTUNE)
    gr_info_dataset = gr_info_dataset.map(transfer_gr_info,num_parallel_calls = tf.data.experimental.AUTOTUNE)
    rpos_info_dataset = rpos_info_dataset.map(transfer_rpos_info,num_parallel_calls = tf.data.experimental.AUTOTUNE)
    rmask_info_dataset = rmask_info_dataset.map(transfer_rmask_info,num_parallel_calls = tf.data.experimental.AUTOTUNE)
    rclass_info_dataset = rclass_info_dataset.map(transfer_rclass_info,num_parallel_calls = tf.data.experimental.AUTOTUNE)

    dataset = tf.data.Dataset.zip((data_dataset,heart_beat_dataset,gr_info_dataset,rpos_info_dataset,
                                   rmask_info_dataset,rclass_info_dataset))#获取所有的数据和数据信息


    def all_transfer(data_dataset,heart_beat_dataset,gr_info_dataset,rpos_info_dataset,rmask_info_dataset,rclass_info_dataset):
        # id=data_dataset[0]
        data=data_dataset[1]
        label=rclass_info_dataset[1]
        # mask=rmask_info_dataset[1]
        # heart_beat=heart_beat_dataset[2]
        # heart_beat_nums=heart_beat_dataset[1]
        # dataset=DS(id=id,features=data,labels=label,hb_labels = heart_beat,hb_nums =heart_beat_nums,rmask=mask )
        # dataset={'features':data,'labels':label,'hb_labels':heart_beat}
        # dataset={'inputs':data,'outputs':label}

        return data,label

    dataset=dataset.map(all_transfer,num_parallel_calls = tf.data.experimental.AUTOTUNE)

    if train:
        dataset=dataset.shuffle(buffer_size = 10000,
         reshuffle_each_iteration = True).batch(batch_size).prefetch(buffer_size = tf.data.experimental.AUTOTUNE)
    else:
        dataset=dataset.batch(batch_size).prefetch(buffer_size = tf.data.experimental.AUTOTUNE)

    return dataset,total_data_size

def load_II(path, batch_size,train=False):
    """
    :param path: 数据的路径
    :param batch_size: 批量大小
    :param train:是否是训练
    :return: tf.dataset
    """



    # 以下是不全是N的数据
    # path = '/deeplearn_data2/experimental_data/191210/Channel_II/NotContainNoise/NonPureN/Train'  # 全部是8个文件
    # 基础数据，片段ID+8sII ，总计2001个float32
    data_path = os.path.join(path, 'Raw/*')
    # 心搏数据，片段ID+心搏数量+心搏类型（12分类）+padding(-1),最长40个数，总计42个数
    heart_beat_path = os.path.join(path, 'Attr/hb_labels*')
    # 干扰数据，片段ID+II导干扰掩码1正常0干扰，所以总点数是1+2000=2001
    gr_info_path = os.path.join(path, 'Attr/hb_noise*')
    # R点位置，片段ID+40个小格[-1,-1,...,位置,...]这种的非零数，总数41
    rpos_info_path = os.path.join(path, 'Attr/Segment40/seq40_RR*')
    # R点掩码，片段ID+40格，有R点的是1，无R点的是0，总数41
    rmask_info_path = os.path.join(path, 'Attr/Segment40/seq40_mask*')
    # 40个小格分类，片段ID+40个小格各个小格的,分类，总计41
    rclass_info_path = os.path.join(path, 'Attr/Segment40/seq40_labels*')
    # 获取文件
    data_files = sorted(tf.io.gfile.glob(data_path))
    heart_beat_files = sorted(tf.io.gfile.glob(heart_beat_path))
    gr_info_files = sorted(tf.io.gfile.glob(gr_info_path))
    rpos_info_files = sorted(tf.io.gfile.glob(rpos_info_path))
    rmask_info_files = sorted(tf.io.gfile.glob(rmask_info_path))
    rclass_info_files = sorted(tf.io.gfile.glob(rclass_info_path))

    def print_info(data_files, size):
        print("[INFO] files:", data_files)
        if len(data_files) < 1:
            print('[ERROR] No train data files found in %s' % path)
            exit(-1)
        total_samples = sum([os.path.getsize(f) / 4 / size for f in data_files])
        return total_samples

    total_data_size = print_info(data_files, 2001)
    total_heart_beat_size = print_info(heart_beat_files, 42)
    total_gr_size = print_info(gr_info_files, 2001)
    total_rpos_size = print_info(rpos_info_files, 41)
    total_rmask_size = print_info(rmask_info_files, 41)
    total_rclass_size = print_info(rclass_info_files, 41)
    assert total_data_size == total_heart_beat_size == total_gr_size == total_rpos_size == total_rmask_size == total_rclass_size
    # 生成dataset
    data_dataset = tf.data.FixedLengthRecordDataset(filenames = data_files, record_bytes = 4 * 2001)
    heart_beat_dataset = tf.data.FixedLengthRecordDataset(filenames = heart_beat_files, record_bytes = 4 * 42)
    gr_info_dataset = tf.data.FixedLengthRecordDataset(filenames = gr_info_files, record_bytes = 4 * 2001)
    rpos_info_dataset = tf.data.FixedLengthRecordDataset(filenames = rpos_info_files, record_bytes = 4 * 41)
    rmask_info_dataset = tf.data.FixedLengthRecordDataset(filenames = rmask_info_files, record_bytes = 4 * 41)
    rclass_info_dataset = tf.data.FixedLengthRecordDataset(filenames = rclass_info_files, record_bytes = 4 * 41)


    def transfer_dataset(value):
        raw_data = tf.io.decode_raw(value, tf.float32)
        id = tf.cast(tf.slice(raw_data, [0], [1]), tf.int32)
        data = tf.reshape(tf.slice(raw_data, [1], [2000]), (1, 2000, 1))  # 四个导联的数据


        return id, data #shape=(1,2000,1)


    def transfer_heart_beat(value):
        raw_data = tf.io.decode_raw(value, tf.float32)
        id = tf.cast(tf.slice(raw_data, [0], [1]), tf.int32)
        heart_beat_num = tf.cast(tf.slice(raw_data, [1], [1]), tf.int32)
        heart_beat = tf.cast(tf.slice(raw_data, [2], [40]),tf.int32)  # 心搏分类

        # heart_beat = tf.reshape(tf.gather(heart_beat, tf.where(tf.math.greater(heart_beat, tf.constant(0)))), [-1])
        # heart_label = tf.reshape(tf.gather(heart_beat, tf.where(tf.math.greater(heart_beat, tf.constant(0)))), [-1])
        # negative_value=tf.reshape(tf.gather(heart_beat, tf.where(tf.math.less(heart_beat, tf.constant(0)))), [-1])
        # heart_beat = tf.concat((tf.constant([12]), heart_label, tf.constant([13]),negative_value),axis = -1)
        return id, heart_beat_num, heart_beat

    def transfer_gr_info(value):
        raw_data = tf.io.decode_raw(value, tf.float32)
        id = tf.cast(tf.slice(raw_data, [0], [1]), tf.int32)
        # 四个导联各导联干扰情况,四行分别写四个导的干扰段数以及起始点
        gr_data = tf.cast(tf.slice(raw_data, [1], [2000]), tf.int32) # 四个导联各导联干扰情况
        return id, gr_data

    def transfer_rpos_info(value):
        raw_data = tf.io.decode_raw(value, tf.float32)
        id = tf.cast(tf.slice(raw_data, [0], [1]), tf.int32)
        # 40个格，在有R点的小格写入r点位置（0-1999），其它的写-1
        rpos_data = tf.cast(tf.slice(raw_data, [1], [40]), tf.int32)
        return id, rpos_data

    def transfer_rmask_info(value):
        raw_data = tf.io.decode_raw(value, tf.float32)
        id = tf.cast(tf.slice(raw_data, [0], [1]), tf.int32)
        # 40个格，在有R点的小格写入1,其它是0
        rmask_data = tf.cast(tf.slice(raw_data, [1], [40]), tf.int32)
        return id, rmask_data

    def transfer_rclass_info(value):
        raw_data = tf.io.decode_raw(value, tf.float32)
        id = tf.cast(tf.slice(raw_data, [0], [1]), tf.int32)
        # 40个格，每个小格的分类
        rclass_data = tf.cast(tf.slice(raw_data, [1], [40]), tf.int32)
        # rclass_data=tf.concat(([12], rclass_data, [13]), axis = -1)
        return id, rclass_data

    data_dataset = data_dataset.map(transfer_dataset,num_parallel_calls = tf.data.experimental.AUTOTUNE)
    heart_beat_dataset = heart_beat_dataset.map(transfer_heart_beat,num_parallel_calls = tf.data.experimental.AUTOTUNE)
    gr_info_dataset = gr_info_dataset.map(transfer_gr_info,num_parallel_calls = tf.data.experimental.AUTOTUNE)
    rpos_info_dataset = rpos_info_dataset.map(transfer_rpos_info,num_parallel_calls = tf.data.experimental.AUTOTUNE)
    rmask_info_dataset = rmask_info_dataset.map(transfer_rmask_info,num_parallel_calls = tf.data.experimental.AUTOTUNE)
    rclass_info_dataset = rclass_info_dataset.map(transfer_rclass_info,num_parallel_calls = tf.data.experimental.AUTOTUNE)

    dataset = tf.data.Dataset.zip((data_dataset,heart_beat_dataset,gr_info_dataset,rpos_info_dataset,
                                   rmask_info_dataset,rclass_info_dataset))#获取所有的数据和数据信息


    def all_transfer(data_dataset,heart_beat_dataset,gr_info_dataset,rpos_info_dataset,rmask_info_dataset,rclass_info_dataset):
        id=data_dataset[0]
        data=data_dataset[1]
        label=rclass_info_dataset[1]
        mask=rmask_info_dataset[1]
        heart_beat=heart_beat_dataset[2]
        heart_beat_nums=heart_beat_dataset[1]
        # dataset=DS(id=id,features=data,labels=label,hb_labels = heart_beat,hb_nums =heart_beat_nums,rmask=mask )
        # dataset={'features':data,'labels':label,'hb_labels':heart_beat}
        # dataset={'inputs':data,'outputs':label}

        return id,data,label,mask,heart_beat,heart_beat_nums

    dataset=dataset.map(all_transfer,num_parallel_calls = tf.data.experimental.AUTOTUNE)

    if train:
        dataset=dataset.shuffle(buffer_size = 10000,
         reshuffle_each_iteration = True).batch(batch_size).prefetch(buffer_size = tf.data.experimental.AUTOTUNE)
    else:
        dataset=dataset.batch(batch_size).prefetch(buffer_size = tf.data.experimental.AUTOTUNE)

    return dataset,total_data_size
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
