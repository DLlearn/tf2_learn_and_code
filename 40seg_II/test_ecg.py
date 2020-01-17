#! /usr/bin/env python
# -*- coding:utf-8 -*-
"""
这是一个用来测试的脚本，需外传模型路径、测式数据路径
"""
import tensorflow as tf
import shutil
import datetime
import matplotlib
matplotlib.use("Agg")#这个设置可以使matplotlib保存.png图到磁盘
import argparse
from load_data import load5 as load
import numpy as np
import pandas as pd
import os
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

def print_metrics(labels, predictions, target_names, save = False, save_path = None):
    # 计算confusion result
    preds = np.argmax(predictions, axis = -1).flatten().tolist()
    assert len(predictions)==len(labels)
    confusion_result = confusion_matrix(labels, preds)
    print("connfusion_result.shape",confusion_result.shape)
    pd.set_option('display.max_rows', 500)

    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1500)
    confusion_result = pd.DataFrame(confusion_result, index = target_names, columns = target_names)
    # classification report
    report = classification_report(labels, preds, target_names = target_names, digits = 4)
    result_report = 'Confuse_matrix:\n{}\n\nClassification_report:\n{} \n'.format(confusion_result, report)
    print(result_report)
    if save:

        savepath = os.path.join(save_path, "predicted_result.txt")

        print('the result saved in %s' % savepath)  # 如果savepath相同的话,会把所有结果保存到同一个文件中

        with open(savepath, 'a') as f:
            f.write(result_report)
def configs(args):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')  # 列出所有可见显卡
    print("All the available GPUs:\n", physical_devices)
    if physical_devices:
        gpu = physical_devices[args.which_gpu]  # 显示第一块显卡
        tf.config.experimental.set_memory_growth(gpu, True)  # 根据需要自动增长显存
        tf.config.experimental.set_visible_devices(gpu, 'GPU')  # 只选择第一块

def prepare_data(args):
    print("test data:", args.test_data_path)
    test_ds, total_test_samples = load(args.test_data_path, args.batch_size, subset = 'test*', train = False)
    print("total_test_samples:", total_test_samples)
    test_steps_per_epoch = np.ceil(total_test_samples / args.batch_size).astype(np.int32)
    print("test_steps_per_epoch:", test_steps_per_epoch)
    return test_ds,test_steps_per_epoch

def arg_parser():
    parser = argparse.ArgumentParser(description = "prepare all the needed parameters")
    parser.add_argument("--test_data_path",type=str,help="test data path")
    parser.add_argument("--model_path",type=str,help="model restored path")
    parser.add_argument("--log_dir",type=str,help="test data path")
    parser.add_argument("--which_gpu",type=int,default=0,help="choise a suitable gpu" )
    parser.add_argument("--batch_size",type=int,default=256,help="training batch size")
    args = parser.parse_args()
    return args


def main():
    args=arg_parser()
    configs(args)
    test_ds,test_steps_per_epoch=prepare_data(args)

    model = tf.keras.models.load_model(args.model_path)
    predictions = model.predict(test_ds, steps = test_steps_per_epoch, verbose = 1)
    print("predictions.shape:", predictions.shape)

    test_labels = []
    for data, label in test_ds:
        test_labels.extend(label.numpy().flatten().tolist())

    print("test_labels.shape", len(test_labels))
    class_names = ['N', 'Af', 'SJ', 'VC', 'SC', 'JC', 'N_CRB', 'N_CLB', 'N_PS', 'Af_CRB', 'N_B1', 'AF']
    print_metrics(test_labels, predictions, class_names, True, args.log_dir)

if __name__=="__main__":
    main()
