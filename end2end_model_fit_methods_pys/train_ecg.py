#! /usr/bin/env python
# -*- coding:utf-8 -*-
"""
本代码训练序列ecg数据，8秒，2000个点，分为40个小块，每个小块50个点
一共有50个标签
II导联，14分类（12分类+干扰+低电压） 格式40+2000
训练数据地址：/deeplearn_data2/experimental_data/191108/train_datas/
测试数据地址：/deeplearn_data2/experimental_data/191108/test_datas/
"""
import tensorflow as tf
import shutil
import datetime
import matplotlib
matplotlib.use("Agg")#这个设置可以使matplotlib保存.png图到磁盘
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
from load_data import load3 as load
from models import model2


def configs(args=None):

    # t = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    t = datetime.datetime.now().strftime("%H%M")
    output_folder= args.output_folder
    if os.path.exists(output_folder):
        inc = input("The model saved path(%s) has exist,Do you want to delete and remake it?(y/n)" % output_folder)
        while (inc.lower() not in ['y', 'n']):
            inc = input("The model saved path has exist,Do you want to delete and remake it?(y/n)")
        if inc.lower() == 'y':
            shutil.rmtree(output_folder)
            os.makedirs(output_folder)
        else:
            print("use the same outout_folder")

    else:
        print("The model saved path (%s) does not exist,make it!"%output_folder)
        os.makedirs(output_folder)

    if args.save_format == "hdf5":
        save_path_models = os.path.join(output_folder, "hdf5_models_{}".format(t))
        if not os.path.exists(save_path_models):
            os.makedirs(save_path_models)
        save_path = os.path.join(save_path_models, "ckpt_epoch{epoch:02d}_val_acc{val_accuracy:.2f}.hdf5")
    elif args.save_format == "saved_model":
        save_path_models = os.path.join(output_folder, "saved_models_{}".format(t))
        if not os.path.exists(save_path_models):
            os.makedirs(save_path_models)
        save_path = os.path.join(save_path_models, "ckpt_epoch{epoch:02d}_val_acc{val_accuracy:.2f}.ckpt")
    # 用来保存日志
    t1 = datetime.datetime.now().strftime("%H%M")
    log_dir = os.path.join(output_folder, 'logs_{}'.format(t1))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    physical_devices = tf.config.experimental.list_physical_devices('GPU')  # 列出所有可见显卡
    print("All the available GPUs:\n", physical_devices)
    if physical_devices:
        gpu = physical_devices[args.which_gpu]  # 显示第一块显卡
        tf.config.experimental.set_memory_growth(gpu, True)  # 根据需要自动增长显存
        tf.config.experimental.set_visible_devices(gpu, 'GPU')  # 只选择第一块
    return output_folder,save_path,log_dir

def prepare_data(args=None):
    print("train data:", args.train_data_path)
    print("test data:", args.validation_data_path)
    train_ds, total_train_samples = load(args.train_data_path, args.batch_size, subset = 'train*', leader = 'II', train = True)
    validation_ds, total_validation_samples = load(args.validation_data_path, args.batch_size, subset = 'test*', leader = 'II', train = False)
    print("total_train_samples:", total_train_samples)
    print("total_test_samples:", total_validation_samples)
    return (train_ds,total_train_samples),(validation_ds,total_validation_samples)

def get_callbacks(args,save_path=None,log_dir=None):
    ckpt = tf.keras.callbacks.ModelCheckpoint(save_path, monitor = 'val_accuracy', verbose = 1,
                                              save_best_only = False, save_weights_only = False,
                                              save_frequency = 1)
    earlystop = tf.keras.callbacks.EarlyStopping(monitor = 'val_accuracy', min_delta = 0.00001, patience = 5,
                                                 verbose = True)

    class LearningRateExponentialDecay:
        def __init__(self, initial_learning_rate, decay_epochs, decay_rate):
            self.initial_learning_rate = initial_learning_rate
            self.decay_epochs = decay_epochs
            self.decay_rate = decay_rate

        def __call__(self, epoch):
            dtype = type(self.initial_learning_rate)
            decay_epochs = np.array(self.decay_epochs).astype(dtype)
            decay_rate = np.array(self.decay_rate).astype(dtype)
            epoch = np.array(epoch).astype(dtype)
            p = np.floor(epoch / decay_epochs)
            lr = self.initial_learning_rate * np.power(decay_rate, p)
            return lr

    lr_schedule = LearningRateExponentialDecay(args.initial_learning_rate, args.lr_decay_epochs, 0.96)
    lr = tf.keras.callbacks.LearningRateScheduler(lr_schedule, verbose = 1)
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir = log_dir)
    terminate = tf.keras.callbacks.TerminateOnNaN()
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss', factor = 0.1, patience = 3, verbose = 1,
                                                     min_delta = 0.0001, min_lr = 0)
    csv_logger = tf.keras.callbacks.CSVLogger(os.path.join(log_dir, 'logs.log'), separator = ',')
    callbacks = [ckpt, earlystop, lr, tensorboard, terminate, reduce_lr, csv_logger]
    return callbacks

def plot_lr(lrs,log_dir,title="Learning Rate Schedule"):
    #计算学习率随epoch的变化值
    epochs=np.arange(len(lrs))
    plt.figure()
    plt.plot(epochs,lrs)
    plt.xticks(epochs)
    plt.scatter(epochs,lrs)
    plt.title(title)
    plt.xlabel("Epoch #")
    plt.ylabel("Learning Rate")
    plt.savefig(os.path.join(log_dir, 'learning_rate.png'))
def plot_acc_loss(history=None,log_dir=None):
    plt.figure()
    N = np.arange(len(history.history['loss']))
    plt.plot(N,history.history['loss'],label='train_loss')
    plt.scatter(N,history.history['loss'])
    plt.plot(N,history.history['val_loss'],label='val_loss')
    plt.scatter(N,history.history['val_loss'])
    plt.plot(N,history.history['accuracy'],label='train_acc')
    plt.scatter(N,history.history['accuracy'])
    plt.plot(N,history.history['val_accuracy'],label='val_acc')
    plt.scatter(N,history.history['val_accuracy'])
    plt.title('Training Loss and Accuracy on Our_dataset')
    plt.xlabel('Epoch #')
    plt.ylabel('Loss/Accuracy')
    plt.legend()
    plt.savefig(os.path.join(log_dir,'training.png'))

def arg_parser():
    parser = argparse.ArgumentParser(description = "prepare all the needed parameters")
    parser.add_argument("--output_folder",type=str,help="all the model saved place")
    parser.add_argument("--train_data_path",type=str,help="train data path")
    parser.add_argument("--validation_data_path",type=str,help="validation data path")
    parser.add_argument("--save_format", type = str, help = "validation data path")
    parser.add_argument("--which_gpu",type=int,default=0,help="choise a suitable gpu" )
    parser.add_argument("--batch_size",type=int,default=256,help="training batch size")
    parser.add_argument("--epochs",type=int,default=60,help="determine the training epochs")
    parser.add_argument("--regularizer",type=float,default=5e-4,help="do parameters regularization")
    parser.add_argument("--lr_decay_epochs",type=int,default=5,help="learning rate decay schedule")
    parser.add_argument("--num_classes",type=int,default = 12,help="class number")
    parser.add_argument("--initial_learning_rate",type=float,default=1e-2,help="initial learning rate")
    args = parser.parse_args()
    return args

def main():

    args = arg_parser()
    #do configuration
    output_folder,save_path,log_dir=configs(args)
    #load data
    (train_ds,total_train_samples),(validation_ds,total_validation_samples) = prepare_data(args)

    #prepare model
    model = model2.model(args.num_classes)
    print(model.summary())
    tf.keras.utils.plot_model(model, to_file = os.path.join(log_dir, 'model.png'), show_shapes = True)
    model_json = model.to_json()
    with open(os.path.join(log_dir, 'model_json.json'), 'w') as json_file:
        json_file.write(model_json)


    optimizer = tf.keras.optimizers.Adam(learning_rate = args.initial_learning_rate)
    # optimizer = tf.keras.optimizers.SGD(learning_rate=initial_learning_rate,momentum=0.95)
    # 损失函数
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)
    metrics = ['accuracy', 'sparse_categorical_crossentropy']
    model.compile(optimizer = optimizer, loss = loss, metrics = metrics)
    #callbacks

    train_steps_per_epoch = int(total_train_samples // args.batch_size)
    test_steps_per_epoch = np.ceil(total_validation_samples / args.batch_size).astype(np.int32)
    print("train_steps_per_epoch:", train_steps_per_epoch)
    print("test_steps_per_epoch:", test_steps_per_epoch)
    callbacks=get_callbacks(args, save_path , log_dir )
    H = model.fit(train_ds, epochs = args.epochs, steps_per_epoch = train_steps_per_epoch, validation_data = validation_ds,
                  validation_steps = test_steps_per_epoch, callbacks = callbacks, verbose = 1)

    plot_lr(H.history['lr'],log_dir)

    plot_acc_loss(H,log_dir)

if __name__=="__main__":
    main()
