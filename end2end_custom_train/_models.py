#! /usr/bin/env python
# -*- coding:utf-8 -*-
import tensorflow as tf


class Conv_Module(tf.keras.models.Model):
    def __init__(self,k,kx,stride,padding='same'):
        super().__init__()
        self.conv=conv1d(k,kx,activation='relu',strides=stride,padding=padding)
    def call(self,inputs):
        x = self.conv(inputs)
        return x
class Inception_Module(tf.keras.models.Model):
    def __init__(self,num3,num7,num11):
        super().__init__()
        self.conv3 = Conv_Module(num3,3,1)
        self.conv7 = Conv_Module(num7,7,1)
        self.conv11 = Conv_Module(num11,11,1)
    def call(self,inputs):
        conv3=self.conv3(inputs)
        conv7=self.conv7(inputs)
        conv11=self.conv11(inputs)
        x=tf.keras.layers.concatenate([conv3,conv7,conv11],axis=-1)
        return x
class Downsample_Module(tf.keras.models.Model):
    def __init__(self,k):
        super().__init__()
        self.conv3=Conv_Module(k,3,2,padding='valid')
        self.pool = maxpool(3,2)
    def call(self,inputs):
        conv3=self.conv3(inputs)
        pool =self.pool(inputs)
        x = tf.keras.layers.concatenate([conv3,pool],axis=-1)
        return x
class Model1(tf.keras.models.Model):
    """
    查看模型summary()  Model1(14).model().summary()
    """
    def __init__(self, classes):
        super().__init__()
        self.conv1 = Conv_Module(96, 5, 1)
        self.maxpool1 = maxpool(3, 2)
        self.conv2 = Conv_Module(128, 5, 1)
        self.maxpool2 = maxpool(3, 2)
        self.inception_module1 = Inception_Module(32, 32, 32)
        self.inception_module2 = Inception_Module(32, 48, 48)
        self.downsample_module1 = Downsample_Module(128)
        self.inception_module3 = Inception_Module(128, 64, 48)
        self.inception_module4 = Inception_Module(96, 96, 64)
        self.inception_module5 = Inception_Module(80, 96, 80)
        self.inception_module6 = Inception_Module(48, 128, 96)
        self.downsample_module2 = Downsample_Module(96)  # 125
        self.inception_module7 = Inception_Module(160, 128, 96)
        self.inception_module8 = Inception_Module(160, 128, 96)
        self.downsample_module3 = Downsample_Module(128)  # 62
        self.inception_module9 = Inception_Module(128, 96, 128)
        self.inception_module10 = Inception_Module(96, 128, 128)
        self.avg_pool = tf.keras.layers.AveragePooling1D(7, 2)  # 28,384
        self.dropout1 = dropout(0.2)
        self.lstm1 = lstm(384, return_sequences = True)  # `28,384
        self.dropout2 = dropout(0.2)
        self.lstm2 = lstm(384)  # 1，384
        self.dropout3 = dropout(0.2)
        self.repeat = tf.keras.layers.RepeatVector(40)
        self.lstm3 = lstm(384, return_sequences = True)  # 40，384
        self.dropout4 = dropout(0.2)
        self.lstm4 = lstm(256, return_sequences = True)  # 40，256
        self.dropout5 = dropout(0.2)
        self.lstm5 = lstm(192, return_sequences = True)  # 40，192
        self.dropout6 = dropout(0.2)

        self.time_distribute = tf.keras.layers.TimeDistributed(fc(classes), name = "output_node")  # 40,14

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)

        x = self.inception_module1(x)
        x = self.inception_module2(x)
        x = self.downsample_module1(x)
        x = self.inception_module3(x)
        x = self.inception_module4(x)
        x = self.inception_module5(x)
        x = self.inception_module6(x)
        x = self.downsample_module2(x)
        x = self.inception_module7(x)
        x = self.inception_module8(x)
        x = self.downsample_module3(x)
        x = self.inception_module9(x)
        x = self.inception_module10(x)

        x = self.avg_pool(x)
        x = self.dropout1(x)
        x = self.lstm1(x)
        x = self.dropout2(x)
        x = self.lstm2(x)
        x = self.dropout3(x)
        x = self.repeat(x)
        x = self.lstm3(x)
        x = self.dropout4(x)
        x = self.lstm4(x)
        x = self.dropout5(x)
        x = self.lstm5(x)
        x = self.dropout6(x)

        x = self.time_distribute(x)
        return x

    def model(self):
        """
        这是一种functional的定义方式,直接返回我们想要的模型
        通常只要用functional的定义方式就可以，优点是可以直接看model.summary()
        很少有模型无法用functional的方式实现
        不足是想要查看各层需要使用model.layers
        Subclassing定义方式却不同，可以把所有用到的层定义到它的属性中
        缺点是无法用model.summary()查看，不方便看数据变化过程，可用于循环等操作
        """
        x = tf.keras.layers.Input(shape = (2000, 1))
        return tf.keras.models.Model(inputs = [x], outputs = self.call(x))
###############################model2###########################################

#####################################model3###################################
def inception_module(x,num3,num7,num11):
    conv3=conv1d(num3,3)(x)
    conv7=conv1d(num7,7)(x)
    conv11=conv1d(num11,11)(x)
    x =tf.keras.layers.concatenate([conv3,conv7,conv11],axis=-1)
    return x
def down_sample(x,num):
    conv = conv1d(num,3,2)(x)
    pool = maxpool(3,2,padding='same')(x)
    x = tf.keras.layers.concatenate([conv,pool],axis=-1)
    return x

def model3(num_classes):
    input_node = tf.keras.layers.Input(shape=(2000,1))
    x = conv1d(64,5)(input_node)
    x = maxpool(5,2,padding='same')(x)
    x = conv1d(128, 5)(input_node)
    x = maxpool(5, 2, padding = 'same')(x)
    x = inception_module(x,64,32,16)
    x = down_sample(x,32)
    x = inception_module(x,64,32,16)
    x = inception_module(x,64,32,16)
    x = inception_module(x,64,32,32)
    x = down_sample(x,32)
    x = inception_module(x,64,32,32)
    x = inception_module(x,128,64,32)
    x = inception_module(x,64,64,64)
    x = down_sample(x,64)
    x = inception_module(x,128,64,64)
    x = down_sample(x,64)
    x = inception_module(x,64,32,32)
    x = down_sample(x,64)
    x = lstm(256,return_sequences=True)(x)
    x = lstm(128,return_sequences=True)(x)
    x = lstm(64)(x)
    x = dropout(0.5)(x)
    x = tf.keras.layers.RepeatVector(40)(x)
    x = lstm(128,return_sequences=True)(x)
    x = lstm(128,return_sequences=True)(x)
    x = tf.keras.layers.TimeDistributed(fc(128))(x)
    x = dropout(0.5)(x)
    x = fc(num_classes,activation=None)(x)
    model=tf.keras.models.Model(inputs=[input_node],outputs=[x])
    return model