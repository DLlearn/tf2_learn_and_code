#! /usr/bin/env python
# -*- coding:utf-8 -*-
import tensorflow as tf

class ObjDict(dict):
    """Makes a dictionary behave like an object, with attribute-style access.
    """
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)
    def __setattr__(self, name, value):
        self[name] = value

def moving_average(model,decay,shadow_variables):
    """
    :param model: tensorflow2.0 models
    :return:
    """
    variables =[decay * shadow_variables[i] + (1 - decay) * model.trainable_variables[i] for i in range(len(model.trainable_variables))]
    model.set_weights(variables)