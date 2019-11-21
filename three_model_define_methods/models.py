#导入必备包
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import  Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import concatenate


#开始定义模型
def shallownet_sequential(width,height,depth,classes):
    #channel last，用输入形状来初始化模型
    model = Sequential()
    inputshape=(height,width,depth)
    model.add(Conv2D(32,(3,3),padding='same',input_shape=inputshape))
    model.add(Activation("relu"))
    
    #softmax
    model.add(Flatten())
    model.add(Dense(classes))
    model.add(Activation("softmax"))
    return model

def conv_module(x,k,kx,ky,stride,chandim,padding="same"):
    #conv-bn-relu
    x = Conv2D(k,(kx,ky),strides=stride,padding=padding)(x)
    x = BatchNormalization(axis=chandim)(x)
    x = Activation('relu')(x)
    return x
def inception_module(x,num1x1,num3x3,chandim):
    conv_1x1 = conv_module(x,num1x1,1,1,(1,1),chandim)
    conv_3_3 = conv_module(x,num3x3,3,3,(1,1),chandim)
    x = concatenate([conv_1x1,conv_3_3],axis=chandim)
    return x
def downsample_module(x,k,chandim):
    #conv downsample and pool downsample
    conv_3x3 = conv_module(x,k,3,3,(2,2),chandim,padding='valid')
    pool = MaxPooling2D((3,3),strides=(2,2))(x)
    x = concatenate([conv_3x3,pool],axis=chandim)
    return x
#然后定义整个MiniGoogLeNet
def minigooglenet_functional(width,height,depth,classes):
    inputshape=(height,width,depth)
    chandim=-1
    #define inputs and firse conv
    inputs = Input(shape=inputshape)
    x = conv_module(inputs,96,3,3,(1,1),chandim)
    #def two inception and followed by a downsample
    x = inception_module(x,32,32,chandim)
    x = inception_module(x,32,48,chandim)
    x = downsample_module(x,80,chandim)
    #def four inception and one downsample
    x = inception_module(x,112,48,chandim)
    x = inception_module(x,96,64,chandim)
    x = inception_module(x,80,80,chandim)
    x = inception_module(x,48,96,chandim)
    x = downsample_module(x,96,chandim)
    #def two inception followed by global pool and dropout
    x = inception_module(x,176,160,chandim)
    x = inception_module(x,176,160,chandim)
    x = AveragePooling2D((7,7))(x)
    x = Dropout(0.5)(x)
    #softmax
    x = Flatten()(x)
    x = Dense(classes)(x)
    x = Activation('softmax')(x)

    #create the model
    model = Model(inputs,x,name='MiniGoogLeNet')
    return model

class MiniVGGNetModel(Model):
    def __init__(self,classes,chandim=-1):
        super().__init__()
        #(conv+relu)*2+pool
        self.conv1a = Conv2D(32,(3,3),padding='same')
        self.act1a = Activation('relu')
        self.bn1a = BatchNormalization(axis=chandim)
        self.conv1b = Conv2D(32,(3,3),padding='same')
        self.act1b = Activation('relu')
        self.bn1b = BatchNormalization(axis=chandim)
        self.pool1 = MaxPooling2D(pool_size=(2,2))
        ##(conv+relu)*2+pool
        self.conv2a = Conv2D(32,(3,3),padding='same')
        self.act2a = Activation('relu')
        self.bn2a = BatchNormalization(axis=chandim)
        self.conv2b = Conv2D(32,(3,3),padding='same')
        self.act2b = Activation('relu')
        self.bn2b = BatchNormalization(axis=chandim)
        self.pool2 = MaxPooling2D(pool_size=(2,2))
        #fully-connected
        self.flatten=Flatten()
        self.dense3=Dense(512)
        self.act3 = Activation('relu')
        self.bn3 = BatchNormalization()
        self.do3 = Dropout(0.5)
        #softmax classifier
        self.dense4=Dense(classes)
        self.softmax=Activation('softmax')
    def call(self,inputs):
        #build 1
        x = self.conv1a(inputs)
        x = self.act1a(x)
        x = self.bn1a(x)
        x = self.conv1b(x)
        x = self.act1b(x)
        x = self.bn1b(x)
        x = self.pool1(x)
        #bulid 2
        x = self.conv2a(x)
        x = self.act2a(x)
        x = self.bn2a(x)
        x = self.conv2b(x)
        x = self.act2b(x)
        x = self.bn2b(x)
        x = self.pool2(x)
        #build fully connected
        x = self.flatten(x)
        x = self.dense3(x)
        x = self.act3(x)
        x = self.do3(x)
        #build softmax
        x = self.dense4(x)
        x = self.softmax(x)
        #return the constructed model
        return x

