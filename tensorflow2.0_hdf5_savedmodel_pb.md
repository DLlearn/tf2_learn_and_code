## 1、训练模型

我们将把训练好的模型分别保存成hdf5和saved model格式，然后完成它们之间的互相转换以及分别转tensorflow1.x的pb格式，具体有：
1. hdf5转saved model,并验证转换后的saved model与直接保存的saved model的无差异性（大小，精度）
2. saved model转hdf5,并验证转换后的hdf5与直接保存的hdf5的无差异性（大小，精度）
3. hdf5转pb,并验证转换后的pb与直接原始的的hdf5的无差异性（大小，精度）
4. saved mode转pb,并验证转换后的pb与直接原始的的saved mode的无差异性（大小，精度）
5. 对比hdf5所转pb与saved model所转pb的区别


```python
import tensorflow as tf
import os
from functools import partial
import numpy as np
import shutil
print("tf.__version__")
```

    tf.__version__



```python
batch_size=64
epochs=6
regularizer=1e-3
total_train_samples=60000
total_test_samples=10000

output_folder="/tmp/test/hdf5_model"
output_folder1="/tmp/test/saved_model"
output_folder2="/tmp/test/pb_model"
for m in (output_folder,output_folder1,output_folder2):
    if os.path.exists(m):
        inc=input("The model(%s) saved path has exist,Do you want to delete and remake it?(y/n)"%m)
        while(inc.lower() not in ['y','n']):
            inc=input("The model saved path has exist,Do you want to delete and remake it?(y/n)")
        if inc.lower()=='y':
            shutil.rmtree(m)
            os.makedirs(m)
    elif not os.path.exists(m):
        os.makedirs(m)

```

    The model(/tmp/test/hdf5_model) saved path has exist,Do you want to delete and remake it?(y/n)y
    The model(/tmp/test/saved_model) saved path has exist,Do you want to delete and remake it?(y/n)y



```python
#指定显卡
physical_devices = tf.config.experimental.list_physical_devices('GPU')#列出所有可见显卡
print("All the available GPUs:\n",physical_devices)
if physical_devices:
    gpu=physical_devices[0]#显示第一块显卡
    tf.config.experimental.set_memory_growth(gpu, True)#根据需要自动增长显存
    tf.config.experimental.set_visible_devices(gpu, 'GPU')#只选择第一块
```

    All the available GPUs:
     [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]



```python
#准备数据
fashion_mnist=tf.keras.datasets.fashion_mnist
(train_x,train_y),(test_x,test_y)=fashion_mnist.load_data()

train_x,test_x = train_x[...,np.newaxis]/255.0,test_x[...,np.newaxis]/255.0
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat','Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
train_ds = tf.data.Dataset.from_tensor_slices((train_x,train_y))
test_ds = tf.data.Dataset.from_tensor_slices((test_x,test_y))
 
train_ds=train_ds.shuffle(buffer_size=batch_size*10).batch(batch_size).prefetch(buffer_size = tf.data.experimental.AUTOTUNE).repeat()
test_ds = test_ds.batch(batch_size).prefetch(buffer_size = tf.data.experimental.AUTOTUNE)#不加repeat，执行一次就行
```


```python
#定义模型
l2 = tf.keras.regularizers.l2(regularizer)#定义模型正则化方法
ini = tf.keras.initializers.he_normal()#定义参数初始化方法
conv2d = partial(tf.keras.layers.Conv2D,activation='relu',padding='same',kernel_regularizer=l2,bias_regularizer=l2)
fc = partial(tf.keras.layers.Dense,activation='relu',kernel_regularizer=l2,bias_regularizer=l2)
maxpool=tf.keras.layers.MaxPooling2D
dropout=tf.keras.layers.Dropout
def test_model():
    x_input = tf.keras.layers.Input(shape=(28,28,1),name='input_node')
    x = conv2d(128,(5,5))(x_input)
    x = maxpool((2,2))(x)
    x = conv2d(256,(5,5))(x)
    x = maxpool((2,2))(x)
    x = tf.keras.layers.Flatten()(x)
    x = fc(128)(x)
    x_output=fc(10,activation=None,name='output_node')(x)
    model = tf.keras.models.Model(inputs=x_input,outputs=x_output) 
    return model
model = test_model()
print(model.summary())
```

    Model: "model"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input_node (InputLayer)      [(None, 28, 28, 1)]       0         
    _________________________________________________________________
    conv2d (Conv2D)              (None, 28, 28, 128)       3328      
    _________________________________________________________________
    max_pooling2d (MaxPooling2D) (None, 14, 14, 128)       0         
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, 14, 14, 256)       819456    
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, 7, 7, 256)         0         
    _________________________________________________________________
    flatten (Flatten)            (None, 12544)             0         
    _________________________________________________________________
    dense (Dense)                (None, 128)               1605760   
    _________________________________________________________________
    output_node (Dense)          (None, 10)                1290      
    =================================================================
    Total params: 2,429,834
    Trainable params: 2,429,834
    Non-trainable params: 0
    _________________________________________________________________
    None



```python
#编译模型
initial_learning_rate=0.01

optimizer = tf.keras.optimizers.SGD(learning_rate=initial_learning_rate,momentum=0.95)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metrics=['accuracy','sparse_categorical_crossentropy']
model.compile(optimizer=optimizer,loss=loss,metrics=metrics)
```


```python
#训练模型
H=model.fit(train_ds,epochs=6,
            steps_per_epoch=np.floor(len(train_x)/batch_size).astype(np.int32),
            validation_data=test_ds,
            validation_steps=np.ceil(len(test_x)/batch_size).astype(np.int32),
            verbose=1)
```

    Train for 937 steps, validate for 157 steps
    Epoch 1/6
    937/937 [==============================] - 78s 84ms/step - loss: 0.9485 - accuracy: 0.8000 - sparse_categorical_crossentropy: 1.5262 - val_loss: 0.6896 - val_accuracy: 0.8527 - val_sparse_categorical_crossentropy: 1.3110
    Epoch 2/6
    937/937 [==============================] - 76s 82ms/step - loss: 0.5635 - accuracy: 0.8824 - sparse_categorical_crossentropy: 1.2657 - val_loss: 0.5024 - val_accuracy: 0.8892 - val_sparse_categorical_crossentropy: 1.2297
    Epoch 3/6
    937/937 [==============================] - 76s 81ms/step - loss: 0.4517 - accuracy: 0.8971 - sparse_categorical_crossentropy: 1.2169 - val_loss: 0.4438 - val_accuracy: 0.8928 - val_sparse_categorical_crossentropy: 1.1814
    Epoch 4/6
    937/937 [==============================] - 76s 81ms/step - loss: 0.4077 - accuracy: 0.9015 - sparse_categorical_crossentropy: 1.1756 - val_loss: 0.4136 - val_accuracy: 0.8995 - val_sparse_categorical_crossentropy: 1.1607
    Epoch 5/6
    937/937 [==============================] - 76s 81ms/step - loss: 0.3850 - accuracy: 0.9051 - sparse_categorical_crossentropy: 1.1504 - val_loss: 0.3992 - val_accuracy: 0.8978 - val_sparse_categorical_crossentropy: 1.1329
    Epoch 6/6
    937/937 [==============================] - 76s 81ms/step - loss: 0.3719 - accuracy: 0.9093 - sparse_categorical_crossentropy: 1.1388 - val_loss: 0.4391 - val_accuracy: 0.8844 - val_sparse_categorical_crossentropy: 1.1219



```python
#分别保存两种格式的模型
model.save(filepath=os.path.join(output_folder,'hdf5_model.h5'),save_format='h5')
model.save(filepath=output_folder1,save_format='tf')
#报的warning信息在tensorflow官网的例子中同样存在
```

    INFO:tensorflow:Assets written to: /tmp/test/saved_model/assets



```bash
%%bash
echo -e "hdf5 model information...\n"
tree "/tmp/test/hdf5_model"

du -ah "/tmp/test/hdf5_model"

echo -e "saved model information...\n"

tree "/tmp/test/saved_model"

du -ah "/tmp/test/saved_model"
```

    hdf5 model information...
    
    /tmp/test/hdf5_model
    └── hdf5_model.h5
    
    0 directories, 1 file
    19M	/tmp/test/hdf5_model/hdf5_model.h5
    19M	/tmp/test/hdf5_model
    saved model information...
    
    /tmp/test/saved_model
    ├── assets
    ├── saved_model.pb
    └── variables
        ├── variables.data-00000-of-00002
        ├── variables.data-00001-of-00002
        └── variables.index
    
    2 directories, 4 files
    4.0K	/tmp/test/saved_model/assets
    216K	/tmp/test/saved_model/saved_model.pb
    19M	/tmp/test/saved_model/variables/variables.data-00001-of-00002
    4.0K	/tmp/test/saved_model/variables/variables.index
    8.0K	/tmp/test/saved_model/variables/variables.data-00000-of-00002
    19M	/tmp/test/saved_model/variables
    19M	/tmp/test/saved_model



```python
#选取第一个样本来做为测试样本,用来评估不同模型之间的精度
test_sample = train_x[0:1]
test_y=train_y[0]
out = model.predict(test_sample)
print("probs:",out[0])
print("true label:{} pred label:{}".format(test_y,np.argmax(out)))

```

    probs: [-2.0793445e+00 -2.2612031e+00 -1.8440809e+00 -1.1460640e+00
     -1.9762940e+00  8.9537799e-03 -3.4592066e+00  3.3828874e+00
      1.5507856e-01  9.2633562e+00]
    true label:9 pred label:9


## 2、各种模型间互转并验证

### 2.1 hdf5转saved model


```python
rm -rf /tmp/test/hdf52saved_model
```


```python
ls /tmp/test/hdf5_model
```

    hdf5_model.h5



```python
tf.keras.backend.clear_session()
hdf5_model = tf.keras.models.load_model("/tmp/test/hdf5_model/hdf5_model.h5")
hdf5_model.save("/tmp/test/hdf52saved_model",save_format='tf')
```

    INFO:tensorflow:Assets written to: /tmp/test/hdf52saved_model/assets



```bash
%%bash
echo -e "hdf5 model information...\n"
tree "/tmp/test/hdf5_model"

du -ah "/tmp/test/hdf5_model"

echo -e "saved model information...\n"

tree "/tmp/test/saved_model"

du -ah "/tmp/test/saved_model"

echo -e "saved model information...\n"

tree "/tmp/test/hdf52saved_model"

du -ah "/tmp/test/hdf52saved_model"
```

    hdf5 model information...
    
    /tmp/test/hdf5_model
    └── hdf5_model.h5
    
    0 directories, 1 file
    19M	/tmp/test/hdf5_model/hdf5_model.h5
    19M	/tmp/test/hdf5_model
    saved model information...
    
    /tmp/test/saved_model
    ├── assets
    ├── saved_model.pb
    └── variables
        ├── variables.data-00000-of-00002
        ├── variables.data-00001-of-00002
        └── variables.index
    
    2 directories, 4 files
    4.0K	/tmp/test/saved_model/assets
    216K	/tmp/test/saved_model/saved_model.pb
    19M	/tmp/test/saved_model/variables/variables.data-00001-of-00002
    4.0K	/tmp/test/saved_model/variables/variables.index
    8.0K	/tmp/test/saved_model/variables/variables.data-00000-of-00002
    19M	/tmp/test/saved_model/variables
    19M	/tmp/test/saved_model
    saved model information...
    
    /tmp/test/hdf52saved_model
    ├── assets
    ├── saved_model.pb
    └── variables
        ├── variables.data-00000-of-00002
        ├── variables.data-00001-of-00002
        └── variables.index
    
    2 directories, 4 files
    4.0K	/tmp/test/hdf52saved_model/assets
    224K	/tmp/test/hdf52saved_model/saved_model.pb
    8.0K	/tmp/test/hdf52saved_model/variables/variables.data-00001-of-00002
    4.0K	/tmp/test/hdf52saved_model/variables/variables.index
    19M	/tmp/test/hdf52saved_model/variables/variables.data-00000-of-00002
    19M	/tmp/test/hdf52saved_model/variables
    19M	/tmp/test/hdf52saved_model


### 2.2 saved model转hdf5


```python
tf.keras.backend.clear_session()
saved_model = tf.keras.models.load_model("/tmp/test/saved_model")
saved_model.save("/tmp/test/hdf5_model/saved2hdf5_model.h5",save_format='h5')
```


    ---------------------------------------------------------------------------

    NotImplementedError                       Traceback (most recent call last)

    <ipython-input-144-322a2edc9a6e> in <module>
          1 tf.keras.backend.clear_session()
          2 saved_model = tf.keras.models.load_model("/tmp/test/saved_model")
    ----> 3 saved_model.save("/tmp/test/hdf5_model/saved2hdf5_model.h5",save_format='h5')
    

    ~/miniconda3/envs/tf2/lib/python3.6/site-packages/tensorflow_core/python/keras/engine/network.py in save(self, filepath, overwrite, include_optimizer, save_format, signatures, options)
        973     """
        974     saving.save_model(self, filepath, overwrite, include_optimizer, save_format,
    --> 975                       signatures, options)
        976 
        977   def save_weights(self, filepath, overwrite=True, save_format=None):


    ~/miniconda3/envs/tf2/lib/python3.6/site-packages/tensorflow_core/python/keras/saving/save.py in save_model(model, filepath, overwrite, include_optimizer, save_format, signatures, options)
        103         not isinstance(model, sequential.Sequential)):
        104       raise NotImplementedError(
    --> 105           'Saving the model to HDF5 format requires the model to be a '
        106           'Functional model or a Sequential model. It does not work for '
        107           'subclassed models, because such models are defined via the body of '


    NotImplementedError: Saving the model to HDF5 format requires the model to be a Functional model or a Sequential model. It does not work for subclassed models, because such models are defined via the body of a Python method, which isn't safely serializable. Consider saving to the Tensorflow SavedModel format (by setting save_format="tf") or using `save_weights`.


以上信息说明，从saved model是无法转换成hdf5模型的，所以个人感觉在训练过程中保存hdf5格式的模型比较好

### 2.3 所有模型精度测试

测试三个模型的精度，原始hdf5模型，原始saved model，hdf5转换的saved model


```python
origin_hdf5_model = tf.keras.models.load_model("/tmp/test/hdf5_model/hdf5_model.h5")
origin_saved_model = tf.keras.models.load_model("/tmp/test/saved_model")
converted_saved_model = tf.keras.models.load_model("/tmp/test/hdf52saved_model")
```


```python
out1 = origin_hdf5_model.predict(test_sample)
out2 = origin_saved_model.predict(test_sample)
out3 = converted_saved_model.predict(test_sample)
print("probs:",out1[0])
print("true label:{} pred label:{}".format(test_y,np.argmax(out1)))
print("probs:",out2[0])
print("true label:{} pred label:{}".format(test_y,np.argmax(out2)))
print("probs:",out3[0])
print("true label:{} pred label:{}".format(test_y,np.argmax(out3)))
np.testing.assert_array_almost_equal(out,out1)
np.testing.assert_array_almost_equal(out,out2)
np.testing.assert_array_almost_equal(out,out3)
```

    probs: [-2.0793445e+00 -2.2612031e+00 -1.8440809e+00 -1.1460640e+00
     -1.9762940e+00  8.9537799e-03 -3.4592066e+00  3.3828874e+00
      1.5507856e-01  9.2633562e+00]
    true label:9 pred label:9
    probs: [-2.0793445e+00 -2.2612031e+00 -1.8440809e+00 -1.1460640e+00
     -1.9762940e+00  8.9537799e-03 -3.4592066e+00  3.3828874e+00
      1.5507856e-01  9.2633562e+00]
    true label:9 pred label:9
    probs: [-2.0793445e+00 -2.2612031e+00 -1.8440809e+00 -1.1460640e+00
     -1.9762940e+00  8.9537799e-03 -3.4592066e+00  3.3828874e+00
      1.5507856e-01  9.2633562e+00]
    true label:9 pred label:9


可以看到结果完全一致，模型转换没有问题

### 2.4 hdf5和saved模型转tensorflow1.x pb模型

我们需要在tensorflow2.0中使用tensorflow1.x内容

以下是hdf5转pb模型


```python
import tensorflow.compat.v1 as tf1
tf1.reset_default_graph()
tf1.keras.backend.set_learning_phase(0) #调用模型前一定要执行该命令
tf1.disable_v2_behavior() #禁止tensorflow2.0的行为
#加载hdf5模型
hdf5_pb_model = tf.keras.models.load_model("/tmp/test/hdf5_model/hdf5_model.h5")
def freeze_session(session,keep_var_names=None,output_names=None,clear_devices=True):
    graph = session.graph
    with graph.as_default():
#         freeze_var_names = list(set(v.op.name for v in tf1.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
#         output_names += [v.op.name for v in tf1.global_variables()]
        print("output_names",output_names)
        input_graph_def = graph.as_graph_def()
#         for node in input_graph_def.node:
#             print('node:', node.name)
        print("len node1",len(input_graph_def.node))
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph =  tf1.graph_util.convert_variables_to_constants(session, input_graph_def,
                                                      output_names)
        
        outgraph = tf1.graph_util.remove_training_nodes(frozen_graph)#云掉与推理无关的内容
        print("##################################################################")
        for node in outgraph.node:
            print('node:', node.name)
        print("len node1",len(outgraph.node))
        return outgraph

frozen_graph = freeze_session(tf1.keras.backend.get_session(),output_names=[out.op.name for out in hdf5_pb_model.outputs])
tf1.train.write_graph(frozen_graph, output_folder2, "hdf52pb.pb", as_text=False)
```

    output_names ['output_node/BiasAdd']
    len node1 626
    INFO:tensorflow:Froze 8 variables.
    INFO:tensorflow:Converted 8 variables to const ops.
    ##################################################################
    node: input_node
    node: conv2d/kernel
    node: conv2d/bias
    node: conv2d/Conv2D
    node: conv2d/BiasAdd
    node: conv2d/Relu
    node: max_pooling2d/MaxPool
    node: conv2d_1/kernel
    node: conv2d_1/bias
    node: conv2d_1/Conv2D
    node: conv2d_1/BiasAdd
    node: conv2d_1/Relu
    node: max_pooling2d_1/MaxPool
    node: flatten/Reshape/shape
    node: flatten/Reshape
    node: dense/kernel
    node: dense/bias
    node: dense/MatMul
    node: dense/BiasAdd
    node: dense/Relu
    node: output_node/kernel
    node: output_node/bias
    node: output_node/MatMul
    node: output_node/BiasAdd
    len node1 24





    '/tmp/test/pb_model/hdf52pb.pb'



以下是saved model转pb模型


```python
import tensorflow.compat.v1 as tf1
tf1.reset_default_graph()
tf1.keras.backend.set_learning_phase(0) #调用模型前一定要执行该命令
tf1.disable_v2_behavior() #禁止tensorflow2.0的行为
#加载hdf5模型
saved_pb_model = tf.keras.models.load_model("/tmp/test/saved_model")
def freeze_session(session,keep_var_names=None,output_names=None,clear_devices=True):
    graph = session.graph
    with graph.as_default():
#         freeze_var_names = list(set(v.op.name for v in tf1.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
#         output_names += [v.op.name for v in tf1.global_variables()]
        print("output_names",output_names)
        input_graph_def = graph.as_graph_def()
#         for node in input_graph_def.node:
#             print('node:', node.name)
        print("len node1",len(input_graph_def.node))
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph =  tf1.graph_util.convert_variables_to_constants(session, input_graph_def,
                                                      output_names)
        
        outgraph = tf1.graph_util.remove_training_nodes(frozen_graph)#云掉与推理无关的内容
        print("##################################################################")
        for node in outgraph.node:
            print('node:', node.name)
        print("len node1",len(outgraph.node))
        return outgraph

frozen_graph = freeze_session(tf1.keras.backend.get_session(),output_names=[out.op.name for out in saved_pb_model.outputs])
tf1.train.write_graph(frozen_graph, output_folder2, "saved2pb.pb", as_text=False)
```

    output_names ['model/StatefulPartitionedCall']
    len node1 304
    INFO:tensorflow:Froze 8 variables.
    INFO:tensorflow:Converted 8 variables to const ops.
    ##################################################################
    node: conv2d/kernel
    node: conv2d/bias
    node: conv2d_1/kernel
    node: conv2d_1/bias
    node: dense/kernel
    node: dense/bias
    node: output_node/kernel
    node: output_node/bias
    node: input_1
    node: model/StatefulPartitionedCall
    len node1 10





    '/tmp/test/pb_model/saved2pb.pb'



以下是转换后的saved model转pb


```python
import tensorflow.compat.v1 as tf1
tf1.reset_default_graph()
tf1.keras.backend.set_learning_phase(0) #调用模型前一定要执行该命令
tf1.disable_v2_behavior() #禁止tensorflow2.0的行为
#加载hdf5模型
hdf52saved_pb_model = tf.keras.models.load_model("/tmp/test/hdf52saved_model/")
def freeze_session(session,keep_var_names=None,output_names=None,clear_devices=True):
    graph = session.graph
    with graph.as_default():
#         freeze_var_names = list(set(v.op.name for v in tf1.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
#         output_names += [v.op.name for v in tf1.global_variables()]
        print("output_names",output_names)
        input_graph_def = graph.as_graph_def()
#         for node in input_graph_def.node:
#             print('node:', node.name)
        print("len node1",len(input_graph_def.node))
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph =  tf1.graph_util.convert_variables_to_constants(session, input_graph_def,
                                                      output_names)
        
        outgraph = tf1.graph_util.remove_training_nodes(frozen_graph)#云掉与推理无关的内容
        print("##################################################################")
        for node in outgraph.node:
            print('node:', node.name)
        print("len node1",len(outgraph.node))
        return outgraph

frozen_graph = freeze_session(tf1.keras.backend.get_session(),output_names=[out.op.name for out in saved_pb_model.outputs])
tf1.train.write_graph(frozen_graph, output_folder2, "hdf52saved2pb.pb", as_text=False)
```

    output_names ['model/StatefulPartitionedCall']
    len node1 304
    INFO:tensorflow:Froze 8 variables.
    INFO:tensorflow:Converted 8 variables to const ops.
    ##################################################################
    node: conv2d/kernel
    node: conv2d/bias
    node: conv2d_1/kernel
    node: conv2d_1/bias
    node: dense/kernel
    node: dense/bias
    node: output_node/kernel
    node: output_node/bias
    node: input_1
    node: model/StatefulPartitionedCall
    len node1 10





    '/tmp/test/pb_model/hdf52saved2pb.pb'




```bash
%%bash
tree /tmp/test

du -ah /tmp/test
```

    /tmp/test
    ├── hdf52saved_model
    │   ├── assets
    │   ├── saved_model.pb
    │   └── variables
    │       ├── variables.data-00000-of-00002
    │       ├── variables.data-00001-of-00002
    │       └── variables.index
    ├── hdf5_model
    │   └── hdf5_model.h5
    ├── pb_model
    │   ├── hdf52pb.pb
    │   ├── hdf52saved2pb.pb
    │   └── saved2pb.pb
    └── saved_model
        ├── assets
        ├── saved_model.pb
        └── variables
            ├── variables.data-00000-of-00002
            ├── variables.data-00001-of-00002
            └── variables.index
    
    8 directories, 12 files
    19M	/tmp/test/hdf5_model/hdf5_model.h5
    19M	/tmp/test/hdf5_model
    9.3M	/tmp/test/pb_model/hdf52saved2pb.pb
    9.3M	/tmp/test/pb_model/saved2pb.pb
    9.3M	/tmp/test/pb_model/hdf52pb.pb
    28M	/tmp/test/pb_model
    4.0K	/tmp/test/saved_model/assets
    216K	/tmp/test/saved_model/saved_model.pb
    19M	/tmp/test/saved_model/variables/variables.data-00001-of-00002
    4.0K	/tmp/test/saved_model/variables/variables.index
    8.0K	/tmp/test/saved_model/variables/variables.data-00000-of-00002
    19M	/tmp/test/saved_model/variables
    19M	/tmp/test/saved_model
    4.0K	/tmp/test/hdf52saved_model/assets
    224K	/tmp/test/hdf52saved_model/saved_model.pb
    8.0K	/tmp/test/hdf52saved_model/variables/variables.data-00001-of-00002
    4.0K	/tmp/test/hdf52saved_model/variables/variables.index
    19M	/tmp/test/hdf52saved_model/variables/variables.data-00000-of-00002
    19M	/tmp/test/hdf52saved_model/variables
    19M	/tmp/test/hdf52saved_model
    84M	/tmp/test


可以看到三个pb模型的大小是相同的，但了节点名称不一样，且打印出来的名称顺序在hdf5模型体现更好

### 2.5 加载并测试pb模型

有三个pb模型分别进行测试


```python
import tensorflow.compat.v1 as tf1
import numpy as np

def load_graph(file_path):
    with tf1.gfile.GFile(file_path,'rb') as f:
        graph_def = tf1.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf1.Graph().as_default() as graph:
        tf1.import_graph_def(graph_def,input_map = None,return_elements = None,name = "",op_dict = None,producer_op_list = None)
    graph_nodes = [n for n in graph_def.node]
    return graph,graph_nodes
```

三个模型依次调用：
- /tmp/test/pb_model/hdf52saved2pb.pb
- /tmp/test/pb_model/saved2pb.pb
- /tmp/test/pb_model/hdf52pb.pb

**第一个模型**


```python
file_path='/tmp/test/pb_model/hdf52pb.pb'
graph,graph_nodes = load_graph(file_path)
print("num nodes",len(graph_nodes))
for node in graph_nodes:
    print('node:', node.name) 

```

    num nodes 24
    node: input_node
    node: conv2d/kernel
    node: conv2d/bias
    node: conv2d/Conv2D
    node: conv2d/BiasAdd
    node: conv2d/Relu
    node: max_pooling2d/MaxPool
    node: conv2d_1/kernel
    node: conv2d_1/bias
    node: conv2d_1/Conv2D
    node: conv2d_1/BiasAdd
    node: conv2d_1/Relu
    node: max_pooling2d_1/MaxPool
    node: flatten/Reshape/shape
    node: flatten/Reshape
    node: dense/kernel
    node: dense/bias
    node: dense/MatMul
    node: dense/BiasAdd
    node: dense/Relu
    node: output_node/kernel
    node: output_node/bias
    node: output_node/MatMul
    node: output_node/BiasAdd



```python
input_node = graph.get_tensor_by_name('input_node:0')
output = graph.get_tensor_by_name('output_node/BiasAdd:0')

config = tf1.ConfigProto()
config.gpu_options.allow_growth = True
# config.gpu_options.per_process_gpu_memory_fraction = 0.25# 设定GPU使用占比
config.gpu_options.visible_device_list = '0'  # '0,1'
config.allow_soft_placement = True
config.log_device_placement = False

with tf1.Session(config=config,graph=graph) as sess:
        logits = sess.run(output, feed_dict = {input_node:test_sample})
print("logits:",logits)
np.testing.assert_array_almost_equal(out,logits)
```

    logits: [[-2.0793445e+00 -2.2612031e+00 -1.8440809e+00 -1.1460640e+00
      -1.9762940e+00  8.9537799e-03 -3.4592066e+00  3.3828874e+00
       1.5507856e-01  9.2633562e+00]]


从以上结果可以看到，hdf5转换的pb结果完全正确

**第二个模型**


```python
file_path='/tmp/test/pb_model/saved2pb.pb'
graph,graph_nodes = load_graph(file_path)
print("num nodes",len(graph_nodes))
for node in graph_nodes:
    print('node:', node.name)
```


    ---------------------------------------------------------------------------

    InvalidArgumentError                      Traceback (most recent call last)

    ~/miniconda3/envs/tf2/lib/python3.6/site-packages/tensorflow_core/python/framework/importer.py in _import_graph_def_internal(graph_def, input_map, return_elements, validate_colocation_constraints, name, op_dict, producer_op_list)
        500         results = c_api.TF_GraphImportGraphDefWithResults(
    --> 501             graph._c_graph, serialized, options)  # pylint: disable=protected-access
        502         results = c_api_util.ScopedTFImportGraphDefResults(results)


    InvalidArgumentError: Input 1 of node model/StatefulPartitionedCall was passed float from conv2d/kernel:0 incompatible with expected resource.

    
    During handling of the above exception, another exception occurred:


    ValueError                                Traceback (most recent call last)

    <ipython-input-108-4ad5ab9ba9bb> in <module>
          1 file_path='/tmp/test/pb_model/saved2pb.pb'
    ----> 2 graph,graph_nodes = load_graph(file_path)
          3 print("num nodes",len(graph_nodes))
          4 for node in graph_nodes:
          5     print('node:', node.name)


    <ipython-input-103-6c7963fc55a7> in load_graph(file_path)
          7         graph_def.ParseFromString(f.read())
          8     with tf1.Graph().as_default() as graph:
    ----> 9         tf1.import_graph_def(graph_def,input_map = None,return_elements = None,name = "",op_dict = None,producer_op_list = None)
         10     graph_nodes = [n for n in graph_def.node]
         11     return graph,graph_nodes


    ~/miniconda3/envs/tf2/lib/python3.6/site-packages/tensorflow_core/python/util/deprecation.py in new_func(*args, **kwargs)
        505                 'in a future version' if date is None else ('after %s' % date),
        506                 instructions)
    --> 507       return func(*args, **kwargs)
        508 
        509     doc = _add_deprecated_arg_notice_to_docstring(


    ~/miniconda3/envs/tf2/lib/python3.6/site-packages/tensorflow_core/python/framework/importer.py in import_graph_def(graph_def, input_map, return_elements, name, op_dict, producer_op_list)
        403       name=name,
        404       op_dict=op_dict,
    --> 405       producer_op_list=producer_op_list)
        406 
        407 


    ~/miniconda3/envs/tf2/lib/python3.6/site-packages/tensorflow_core/python/framework/importer.py in _import_graph_def_internal(graph_def, input_map, return_elements, validate_colocation_constraints, name, op_dict, producer_op_list)
        503       except errors.InvalidArgumentError as e:
        504         # Convert to ValueError for backwards compatibility.
    --> 505         raise ValueError(str(e))
        506 
        507     # Create _DefinedFunctions for any imported functions.


    ValueError: Input 1 of node model/StatefulPartitionedCall was passed float from conv2d/kernel:0 incompatible with expected resource.


可以看出saved model转pb是可以的，但使用还是不可以使用

**另外一种调用hdf5转换的pb(或在tensorflow2.x中调用tensorflow1.x转的pb)**


```python
tf1.reset_default_graph()
tf1.enable_v2_behavior()#tensorflow2.x中调用tensorflow1.x的内容需要激活tensorflow2.x的特性
tf.keras.backend.clear_session()
def wrap_frozen_graph(graph_def, inputs, outputs):
    def _imports_graph_def():
        tf1.import_graph_def(graph_def, name="")
    wrapped_import = tf1.wrap_function(_imports_graph_def, [])
    import_graph = wrapped_import.graph
    return wrapped_import.prune(
        tf.nest.map_structure(import_graph.as_graph_element, inputs),
        tf.nest.map_structure(import_graph.as_graph_element, outputs))
```


```python
file_path='/tmp/test/pb_model/hdf52pb.pb'
with open(file_path,'rb') as f:
    graph_def = tf1.GraphDef()
    graph_def.ParseFromString(f.read())
    for node in graph_def.node:
        print("node.name",node.name)
```

    node.name input_node
    node.name conv2d/kernel
    node.name conv2d/bias
    node.name conv2d/Conv2D
    node.name conv2d/BiasAdd
    node.name conv2d/Relu
    node.name max_pooling2d/MaxPool
    node.name conv2d_1/kernel
    node.name conv2d_1/bias
    node.name conv2d_1/Conv2D
    node.name conv2d_1/BiasAdd
    node.name conv2d_1/Relu
    node.name max_pooling2d_1/MaxPool
    node.name flatten/Reshape/shape
    node.name flatten/Reshape
    node.name dense/kernel
    node.name dense/bias
    node.name dense/MatMul
    node.name dense/BiasAdd
    node.name dense/Relu
    node.name output_node/kernel
    node.name output_node/bias
    node.name output_node/MatMul
    node.name output_node/BiasAdd



```python
model_func = wrap_frozen_graph(
    graph_def, inputs='input_node:0',
    outputs='output_node/BiasAdd:0')

o=model_func(tf.constant(test_sample,dtype=tf.float32))

print(o)

np.testing.assert_array_almost_equal(out,o.numpy())
```

    tf.Tensor(
    [[-2.0793445e+00 -2.2612031e+00 -1.8440809e+00 -1.1460640e+00
      -1.9762940e+00  8.9537799e-03 -3.4592066e+00  3.3828874e+00
       1.5507856e-01  9.2633562e+00]], shape=(1, 10), dtype=float32)


## 总结

1. tensorflow2.x保存的hdf5模型可以转tensorflow1.x的pb ,也可以转tensorflow2.x saved model
2. saved model可以pb ,但是转换后无法使用
3. 在tensorflow2.x中可以使用tensorflow1.x或tensorflow2.x的语法来调用
所以我们在以后可以只保存hdf5模型


```python

```
