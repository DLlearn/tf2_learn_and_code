# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")#这个设置可以使matplotlib保存.png图到磁盘
from models import MiniVGGNetModel
from models import minigooglenet_functional
from models import shallownet_sequential

import numpy as np

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import cifar10

import matplotlib.pyplot as plt 
import argparse

#构建命令行解析
#有三种模型结构可以选；将模型结构画出来并保存
parser = argparse.ArgumentParser()
parser.add_argument("-m","--model",type=str,default="sequential",choices=["sequential","functional","subclass"],help="type of models")
parser.add_argument("-p","--plot",type=str,required=True,help="path to output plot file")
args=vars(parser.parse_args())


#initialize parameters
init_lr =1e-2
batch_size=128
num_epochs=60
#初始化cifar10的标签名
labelnames=["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
#load cifar10数据
(train_x,train_y),(test_x,test_y)=cifar10.load_data()
train_x = train_x.astype("float32")/255.0
test_x = test_x.astype("float32")/255.0
#将labels从数转成向量（one-hot方式）
lb =LabelBinarizer()
train_y=lb.fit_transform(train_y)
test_y=lb.transform(test_y)
#构建数据扩增
aug = ImageDataGenerator(rotation_range=18,zoom_range=0.15,width_shift_range=0.2,height_shift_range=0.2,shear_range=0.15,
                        horizontal_flip=True,fill_mode="nearest")

#开始生成模型
if args["model"] == "sequential":
    # instantiate a Keras Sequential model
    print(" using sequential model...")
    model = shallownet_sequential(32, 32, 3, len(labelnames))
 
 # check to see if we are using a Keras Functional model
elif args["model"] == "functional":
    # instantiate a Keras Functional model
    print("using functional model...")
    model = minigooglenet_functional(32, 32, 3, len(labelnames))
 
 # check to see if we are using a Keras Model class
elif args["model"] == "class":
    # instantiate a Keras Model sub-class model
    print(" using model sub-classing...")
    model = MiniVGGNetModel(len(labelnames))
    
#编译模型
opt = SGD(lr=init_lr,momentum=0.9,decay=init_lr/num_epochs)
model.compile(optimizer=opt,loss="categorical_crossentropy",metrics=["accuracy"])
print("start training ...")
H=model.fit_generator(aug.flow(train_x,train_y,batch_size=batch_size),
                    validation_data=(test_x,test_y),
                    steps_per_epoch = train_x.shape[0]//batch_size,
                    epochs=num_epochs,
                    verbose=1
                   )


#测试模型
print("start to evaluating network...")
predictions = model.predict(test_x,batch_size=batch_size)
print(classification_report(test_y.argmax(axis=1),
                           predictions.argmax(axis=1),target_names=labelnames))
#画结果图
N = np.arange(0,num_epochs)
title = "Training Loss and Accuracy on CIFAR-10({})".format(
            args["model"])

plt.style.use("ggplot")
plt.figure()
plt.plot(N,H.history["loss"],label="train_loss")
plt.plot(N,H.history["val_loss"],label="val_loss")
plt.plot(N,H.history["accuracy"],label="train_acc")
plt.plot(N,H.history["val_accuracy"],label="val_acc")
plt.title(title)
plt.savefig(args["plot"])
