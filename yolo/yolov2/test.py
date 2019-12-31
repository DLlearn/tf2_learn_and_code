#! /usr/bin/env python
# -*- coding:utf-8 -*-
import pydevd_pycharm
pydevd_pycharm.settrace('192.168.22.215',port=2222,stdoutToServer=True,stderrToServer = True)

a = 6
b = 8