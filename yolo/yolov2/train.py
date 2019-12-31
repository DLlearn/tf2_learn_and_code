#! /usr/bin/env python
# -*- coding:utf-8 -*-
import pydevd_pycharm
pydevd_pycharm.settrace('192.168.22.215',port=2222,stdoutToServer=True,stderrToServer = True)
import argparse
import configparser
import io
import os
from collections import defaultdict


import numpy as np
from tensorflow.keras.layers import (Conv2D, GlobalAveragePooling2D, Input, Lambda,MaxPooling2D,Concatenate,BatchNormalization,LeakyReLU)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import plot_model as plot

parser = argparse.ArgumentParser(
    description='Yet Another Darknet To Keras Converter.')
parser.add_argument('config_path', help='Path to Darknet cfg file.')
parser.add_argument('weights_path', help='Path to Darknet weights file.')
parser.add_argument('output_path', help='Path to output Keras model file.')
parser.add_argument(
    '-p',
    '--plot_model',
    help='Plot generated Keras model and save as image.',
    action='store_true')
parser.add_argument(
    '-flcl',
    '--fully_convolutional',
    help='Model is fully convolutional so set input shape to (None, None, 3). '
    'WARNING: This experimental option does not work properly for YOLO_v2.',
    action='store_true')
def main(args):


if __name__ == '__main__':
    main(parser.parse_args())