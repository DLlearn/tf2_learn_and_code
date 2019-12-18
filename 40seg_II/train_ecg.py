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
import shutil,os,sys,io,copy,time,itertools,argparse,matplotlib
matplotlib.use("Agg")  # 这个设置可以使matplotlib保存.png图到磁盘
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import pandas as pd
from collections import namedtuple
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from losses import sparse_cross_entropy as sce
from losses import sparse_cross_entropy_label_smooth as scels
from miscellaneous import moving_average
from load_data import load5 as load1
from load_data import load_II as load


from models import inception,model1

History = namedtuple('History', ['train_epoch_acc', 'train_epoch_loss', 'val_epoch_acc', 'val_epoch_loss'])


def configs(args = None):
    # t = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    t = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder = args.output_folder
    if os.path.exists(output_folder):
        inc = input("The model saved path(%s) has exist,Do you want to delete and remake it?(y/n)" % output_folder)
        while (inc.lower() not in ['y', 'n']):
            inc = input("The model saved path has exist,Do you want to delete and remake it?(y/n)")
        if inc.lower() == 'y':
            shutil.rmtree(output_folder)
            os.makedirs(output_folder)
        else:
            print("Exit and chechk the path!")
            exit(-1)

    else:
        print("The model saved path (%s) does not exist,make it!" % output_folder)
        os.makedirs(output_folder)

    if args.save_format == "hdf5":
        save_path_models = os.path.join(output_folder, "hdf5_models_{}".format(t))
        if not os.path.exists(save_path_models):
            os.makedirs(save_path_models)
        save_path = os.path.join(save_path_models, "ckpt_epoch{:02d}_val_acc{:.2f}.hdf5")
    elif args.save_format == "saved_model":
        save_path_models = os.path.join(output_folder, "saved_models_{}".format(t))
        if not os.path.exists(save_path_models):
            os.makedirs(save_path_models)
        save_path = os.path.join(save_path_models, "ckpt_epoch{:02d}_val_acc{:.2f}.ckpt")
    # 用来保存日志
    # t1 = datetime.now().strftime("%Y%m%d_%H%M")
    log_dir = os.path.join(output_folder, 'logs_{}'.format(t))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    physical_devices = tf.config.experimental.list_physical_devices('GPU')  # 列出所有可见显卡
    # print("All the available GPUs:\n", physical_devices)
    if physical_devices:
        gpu = physical_devices[args.which_gpu]  # 显示第一块显卡
        tf.config.experimental.set_memory_growth(gpu, True)  # 根据需要自动增长显存
        tf.config.experimental.set_visible_devices(gpu, 'GPU')  # 只选择第一块
    return output_folder, save_path, log_dir


def prepare_data(args = None):
    # print("train data:", args.train_data_path)
    # print("test data:", args.validation_data_path)
    train_ds, total_train_samples = load(args.train_data_path, args.batch_size, train = True)
    validation_ds, total_validation_samples = load(args.validation_data_path, args.batch_size,train = False)
    # train_ds, total_train_samples = load(args.train_data_path, args.batch_size, subset = 'train_0*', train = True,
    #                                      epoch = args.epochs)
    # validation_ds, total_validation_samples = load(args.validation_data_path, args.batch_size, subset = 'test_0*',
    #                                                train = False)
    print("total_train_samples:", int(total_train_samples))
    print("total_test_samples:", int(total_validation_samples))
    return (train_ds, total_train_samples), (validation_ds, total_validation_samples)



def plot_acc_loss1(history = None, log_dir = None):
    plt.figure()
    N = np.arange(len(history.train_epoch_acc))
    plt.plot(N, history.train_epoch_loss, label = 'train_loss')
    plt.scatter(N, history.train_epoch_loss)
    plt.plot(N, history.val_epoch_loss, label = 'val_loss')
    plt.scatter(N, history.val_epoch_loss)
    plt.plot(N, history.train_epoch_acc, label = 'train_acc')
    plt.scatter(N, history.train_epoch_acc)
    plt.plot(N, history.val_epoch_acc, label = 'val_acc')
    plt.scatter(N, history.val_epoch_acc)
    plt.title('Training Loss and Accuracy on Our_dataset')
    plt.xlabel('Epoch #')
    plt.ylabel('Loss/Accuracy')
    plt.legend()
    plt.savefig(os.path.join(log_dir, 'train_history.png'))
def plot_acc_loss(history=None,log_dir=None):
    plt.figure(figsize=(8,8))
    N = np.arange(len(history.train_epoch_acc))
    plt.subplot(2,1,1)
    plt.plot(N, history.train_epoch_acc, label = 'Training Accuracy')
    plt.scatter(N, history.train_epoch_acc)
    plt.plot(N, history.val_epoch_acc, label = 'Validation Accuracy')
    plt.scatter(N, history.val_epoch_acc)
    plt.legend(loc = 'lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()), 1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(N,history.train_epoch_loss,label='Training Loss')
    plt.scatter(N,history.train_epoch_loss)
    plt.plot(N,history.val_epoch_loss,label='Validation Loss')
    plt.scatter(N,history.val_epoch_loss)
    plt.legend(loc = 'upper right')
    plt.ylabel('Cross Entropy')
    # plt.ylim([0, 1.0])
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.savefig(os.path.join(log_dir,'training.png'))


def arg_parser():
    def boolean_string(s):
        if s not in {'False', 'True'}:
            raise ValueError('Not a valid boolean string')
        return s == 'True'
    parser = argparse.ArgumentParser(description = "prepare all the needed parameters")
    parser.add_argument("--output_folder", type = str, help = "all the model saved place")
    parser.add_argument("--train_data_path", type = str, help = "train data path")
    parser.add_argument("--validation_data_path", type = str, help = "validation data path")
    parser.add_argument("--save_format", type = str, help = "validation data path")
    parser.add_argument("--which_gpu", type = int, default = 0, help = "choise a suitable gpu")
    parser.add_argument("--batch_size", type = int, default = 256, help = "training batch size")
    parser.add_argument("--epochs", type = int, default = 60, help = "determine the training epochs")
    parser.add_argument("--regularizer", type = float, default = 5e-4, help = "do parameters regularization")
    parser.add_argument("--num_classes", type = int, default = 12, help = "class number")
    parser.add_argument("--initial_learning_rate", type = float, default = 1e-2, help = "initial learning rate")
    parser.add_argument("--num_epochs_per_decay", type = float, default = 1,
                        help = "Epochs after which learning rate decays.")
    parser.add_argument("--learning_rate_decay_factor", type = float, default = 0.9,
                        help = "Learning rate decay factor.")
    parser.add_argument("--continue_train", type = boolean_string, default = False,
                        help = "Whether do continue training ")
    parser.add_argument("--pretrained_model_path", type = str, default = None, help = "if continue_train is true,"
                                                                                      "this parameter should specified")
    parser.add_argument("--moving_average_decay_factor", type = float, default = 0.99,
                        help = "Moving average decay factor.")

    args = parser.parse_args()
    return args


def print_metrics(labels, predictions, target_names, save = False, save_path = None, epoch = None, train_time = None,
                  test_time = None):
    # 计算confusion result
    assert len(predictions) == len(labels)
    confusion_result = confusion_matrix(labels, predictions)
    pd.set_option('display.max_rows', 500)

    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1500)
    index = (set(predictions) | set(labels))
    target_names = [target_names[i] for i in index]
    confusion_result = pd.DataFrame(confusion_result, index = target_names, columns = target_names)
    # classification report
    report = classification_report(labels, predictions, target_names = target_names, digits = 4)
    result_report = 'Epoch:{} with train_time:{:2f}min and test_time:{:2f}min\n' \
                    'Confuse_matrix:\n{}\n\nClassification_report:\n{} \n'.format(epoch,
                                                                                  train_time / 60,
                                                                                  test_time / 60,
                                                                                  confusion_result,
                                                                                  report)
    print(result_report)
    if save:

        savepath = os.path.join(save_path, "validation_result.txt")

        print('the result saved in %s' % savepath)  # 如果savepath相同的话,会把所有结果保存到同一个文件中

        with open(savepath, 'a') as f:
            f.write(result_report)
    return confusion_result

def draw_variable_tb(variable,name,step):
    """# 将变量做为图来显示到tensorboard,可能看不懂，
    所以正常训练没有必要使用
    """
    v_img = tf.squeeze(variable)
    shape = v_img.shape

    if len(shape) == 1:  # bias case
        v_img = tf.reshape(v_img, [1, shape[0], 1, 1])
    elif len(shape) == 2:  # dense layer的情形
        if shape[0] > shape[1]:
            v_img = tf.transpose(v_img)
            shape = v_img.shape
        v_img = tf.reshape(v_img, [1, shape[0], shape[1], 1])
    elif len(shape) == 3:  # 这种情形应该不会存在
        v_img = tf.reshape(v_img, [shape[2], shape[0], shape[1], 1])
    elif len(shape) == 4:  # conv
        v_img = tf.transpose(v_img, [3, 2, 0, 1])
        shape = v_img.shape
        v_img = tf.reshape(v_img, [shape[0] * shape[1], shape[2], shape[3], 1])

    shape = v_img.shape

    if len(shape) == 4 and shape[-1] in [1, 3, 4]:
        tf.summary.image(name, v_img, max_outputs = 16, step = step)  # 最多输出16张图
def plot_to_image(figure, log_dir, epoch):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    fig = figure
    plt.savefig(buf, format = 'png')
    fig.savefig(os.path.join(log_dir, 'confusion_matrix_epoch%d.png' % epoch))  # 保存图片
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels = 4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image


def plot_confusion_matrix(cm, class_names):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
    cm (array, shape = [n, n]): a confusion matrix of integer classes
    class_names (array, shape = [n]): String names of the integer classes
    """
    figure = plt.figure(figsize = (8, 8))
    plt.imshow(cm, interpolation = 'nearest', cmap = plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation = 45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis = 1)[:, np.newaxis], decimals = 2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm.iloc[i, j] > threshold.iloc[i] else "black"
        plt.text(j, i, cm.iloc[i, j], horizontalalignment = "center", color = color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return figure


def main():
    # 0 = all messages are logged (default behavior)
    # 1 = INFO messages are not printed
    # 2 = INFO and WARNING messages are not printed
    # 3 = INFO, WARNING, and ERROR messages are not printed
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    tf.keras.backend.clear_session()
    args = arg_parser()
    # do configuration
    output_folder, save_path, log_dir = configs(args)
    # use tensorboard
    train_log_dir = os.path.join(log_dir, 'train')
    validation_log_dir = os.path.join(log_dir, 'validation')
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    validation_summary_writer = tf.summary.create_file_writer(validation_log_dir)
    # load data
    (train_ds, total_train_samples), (validation_ds, total_validation_samples) = prepare_data(args)
    train_steps_per_epoch = np.ceil(total_train_samples / args.batch_size).astype(np.int32)
    test_steps_per_epoch = np.ceil(total_validation_samples / args.batch_size).astype(np.int32)
    print("train_steps_per_epoch:", train_steps_per_epoch)
    print("test_steps_per_epoch:", test_steps_per_epoch)

    # prepare model
    if args.continue_train:
    # model = inception.infer(args.num_classes)
        print("Resotre model from %s and continue train." %(args.pretrained_model_path))

        model = tf.keras.models.load_model(args.pretrained_model_path)
        # model=tf.keras.models.Model(inputs=old_model.inputs,outputs=old_model.outputs)
        # model.set_weights(old_model.get_weights())
    else:
    # model = inception.infer(args.num_classes)
        print("Train a new model.")
        model = model1.infer(args.num_classes,'test_model')
    print(model.summary())
    tf.keras.utils.plot_model(model, to_file = os.path.join(log_dir, 'model_arch.png'), show_shapes = True)
    model_json = model.to_json()
    with open(os.path.join(log_dir, 'model_json.json'), 'w') as json_file:
        json_file.write(model_json)

    # optimizer = tf.keras.optimizers.Adam(learning_rate = args.initial_learning_rate)
    decay_steps = int(train_steps_per_epoch * args.num_epochs_per_decay)

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        args.initial_learning_rate,
        decay_steps = decay_steps,
        decay_rate = args.learning_rate_decay_factor,
        staircase = True)
    # optimizer = tf.keras.optimizers.SGD(learning_rate = lr_schedule, momentum = 0.9)
    optimizer = tf.keras.optimizers.Adam(learning_rate = lr_schedule)

    @tf.function
    def train_on_batch(model,optimizer,datas,labels,
                       train_batch_acc,
                       train_batch_total_loss,
                       train_batch_celoss,
                       train_batch_regloss,
                       train_epoch_acc,
                       train_epoch_ce_loss):
        # global model,optimizer
        with tf.GradientTape() as tape:
            logits = model(datas)
            ce_loss = sce.compute_loss(labels, logits)
            # ce_loss = scels.compute_loss(labels, logits,args.num_classes)
            reg_loss = tf.add_n(model.losses)
            # total_loss = ce_loss + reg_loss  # 加正则化
            total_loss = ce_loss   # 不加正则化
        gradients = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        train_batch_acc(labels, logits)
        train_batch_total_loss(total_loss)
        train_batch_celoss(ce_loss)
        train_batch_regloss(reg_loss)
        train_epoch_acc(labels, logits)
        train_epoch_ce_loss(ce_loss)
        return gradients

    @tf.function
    def test_on_batch(model, datas, labels, val_epoch_acc, test_epoch_loss):
        logits = model(datas)
        loss = sce.compute_loss(labels, logits)
        val_epoch_acc(labels, logits)
        test_epoch_loss(loss)
        preds = tf.argmax(logits, axis = -1)
        return preds

    train_batch_acc = tf.keras.metrics.SparseCategoricalAccuracy()
    train_batch_total_loss = tf.keras.metrics.Mean()  # 交叉熵的loss与正则化的loss的和
    train_batch_celoss = tf.keras.metrics.Mean()  # 交叉熵的loss
    train_batch_regloss = tf.keras.metrics.Mean()  # 正则化的loss
    train_epoch_acc = tf.keras.metrics.SparseCategoricalAccuracy()
    train_epoch_ce_loss = tf.keras.metrics.Mean()

    val_epoch_acc = tf.keras.metrics.SparseCategoricalAccuracy()
    val_epoch_loss = tf.keras.metrics.Mean()

    train_acc = []
    train_loss = []
    val_acc = []
    val_loss = []
    format_str = (
        '%s: step:%-6d epoch:%-6.3f/%d celoss:%-5.2f regloss:%-6.4f total_loss:%-6.2f '
        'batch_acc:%-5.2f%% epoch_acc:%-5.2f%% epoch_loss:%-6.2f (%.1f examples/sec; %-4.3f sec/batch)')
    # format_str=('%s:step:%d, epoch:%.4f/%d loss:%.2f lr:%-7.5f train_batch_acc:%5.2f (%.1f examples/sec; %.3f sec/batch)')
    for epoch in range(args.epochs):
        print("Do training Epoch=%d/%d on train dataset>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>:" % (epoch + 1, args.epochs))

        start = time.time()
        for i, (id,data,label,mask,heart_beat,heart_beat_nums) in enumerate(train_ds):
            if (epoch==0 and i==1):  # 只对第二次batch做porfiling
                tf.summary.trace_on(graph = True, profiler = True)  # 开启Trace（可选）可以记录图结构和profile信息,graph=True会把图结构写入log
            start_time = time.time()
            #prepare moving average parameters
            # num_updates= i+epoch*train_steps_per_epoch
            # moving_average_decay = min(args.moving_average_decay_factor, (1 + num_updates) / (10 + num_updates))
            # shadow_variables=copy.deepcopy(model.trainable_variables)
            # updata variable
            grads = train_on_batch(model, optimizer, data, label, train_batch_acc,
                                   train_batch_total_loss,
                                   train_batch_celoss,
                                   train_batch_regloss,
                                   train_epoch_acc,
                                   train_epoch_ce_loss)

            #do moving average
            # moving_average(model,moving_average_decay,shadow_variables)

            duration = time.time() - start_time
            if (epoch==0 and i==1):
                with train_summary_writer.as_default():
                    tf.summary.trace_export(name = "model_trace", step = 1,
                                            profiler_outdir = train_log_dir)  # 保存Trace信息到文件（可选）
                    tf.summary.trace_off()  # 关闭

            if (i + 1) % 50 == 0:
                examples_per_sec = args.batch_size / duration
                current_epoch = (i + 1) / ((epoch + 1) * train_steps_per_epoch) + epoch
                print(format_str % (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), i + 1,
                                    current_epoch, args.epochs,
                                    train_batch_celoss.result(),
                                    train_batch_regloss.result(),
                                    train_batch_total_loss.result(),
                                    100 * train_batch_acc.result(),
                                    100 * train_epoch_acc.result(),
                                    train_epoch_ce_loss.result(),
                                    examples_per_sec, duration))
                step = tf.constant(epoch * train_steps_per_epoch + i + 1)
                with train_summary_writer.as_default():  # 每50步记录一下，太频繁会影响训练速度
                    tf.summary.scalar('train_batch_accuracy', train_batch_acc.result(), step = step)
                    tf.summary.scalar('train_batch_celoss', train_batch_celoss.result(), step = step)
                    tf.summary.scalar('train_batch_regloss', train_batch_regloss.result(), step = step)
                    tf.summary.scalar('train_batch_total_loss', train_batch_total_loss.result(), step = step)
                    tf.summary.scalar('train_epoch_acc', train_epoch_acc.result(), step = step)
                    tf.summary.scalar('train_epoch_ce_loss', train_epoch_ce_loss.result(), step = step)
                    train_summary_writer.flush()

            if ((i + 1) % int(train_steps_per_epoch * 0.1))==0:
                step = tf.constant(epoch * train_steps_per_epoch + i + 1)
                # 每0.1epoch记录一下模型各层参数及其梯度的直方图，太多日志文件会很大
                with train_summary_writer.as_default():
                    for grad, variable in zip(grads, model.trainable_variables):
                        v_name = variable.name.replace(':', '_')
                        # 记录变量直方图
                        tf.summary.histogram(v_name, variable, step = step)
                        # 记录变量梯度直方图
                        tf.summary.histogram('{}_grad'.format(v_name), grad, step = step)
                        #draw_variable_tb(variable, v_name, step)

                    train_summary_writer.flush()

            train_batch_acc.reset_states()
            train_batch_celoss.reset_states()
            train_batch_regloss.reset_states()
            train_batch_total_loss.reset_states()

        end = time.time() - start
        print("Training Epoch:{}/{} loss:{:.4f} acc:{:.4f} fineshed usetime:{:.1f} sec".format(epoch + 1,
                                                                                               args.epochs,
                                                                                               train_epoch_ce_loss.result().numpy(),
                                                                                               train_epoch_acc.result().numpy(),
                                                                                               end))


        train_acc.append(train_epoch_acc.result().numpy())
        train_loss.append(train_epoch_ce_loss.result().numpy())
        train_epoch_acc.reset_states()
        train_epoch_ce_loss.reset_states()
        #完成一个epoch的训练
        train_summary_writer.flush()
        # 对测试集进行测试
        print("Do testing on validation dataset>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>:")
        all_labels = []
        all_preds = []
        start_time = time.time()
        # for i, (data, label) in enumerate(validation_ds.take(1000)):
        for i,(id,data,label,mask,heart_beat,heart_beat_nums) in enumerate(validation_ds):
            preds = test_on_batch(model, data, label, val_epoch_acc, val_epoch_loss)
            all_preds.extend(preds.numpy().flatten().tolist())
            all_labels.extend(label.numpy().flatten().tolist())
            sys.stdout.write('\r %d / %d finished !' %(i+1,test_steps_per_epoch))
        duration = time.time() - start_time
        print("Epoch %d: test_acc:%.3f test_loss:%.3f  total_time:%d sec " % ((epoch + 1),
                                                                             val_epoch_acc.result(),
                                                                             val_epoch_loss.result(),
                                                                             int(duration)))
        with validation_summary_writer.as_default():
            tf.summary.scalar('val_loss', val_epoch_loss.result(), step = epoch)
            tf.summary.scalar('val_accuracy', val_epoch_acc.result(), step = epoch)
            validation_summary_writer.flush()

        acc = val_epoch_acc.result().numpy()
        val_acc.append(val_epoch_acc.result().numpy())
        val_loss.append(val_epoch_loss.result().numpy())
        class_names = ['N', 'Af', 'SJ', 'VC', 'SC', 'JC', 'N_CRB', 'N_CLB', 'N_PS', 'Af_CRB', 'N_B1', 'AF']
        cm = print_metrics(all_labels, all_preds, class_names, True, validation_log_dir, (epoch + 1), train_time = end,
                           test_time = duration)
        figure = plot_confusion_matrix(cm, class_names = class_names)
        cm_image = plot_to_image(figure, validation_log_dir, epoch + 1)  # 同时保存图片到文件夹
        with validation_summary_writer.as_default():
            tf.summary.scalar('val_loss', val_epoch_loss.result(), step = epoch + 1)
            tf.summary.scalar('val_accuracy', val_epoch_acc.result(), step = epoch + 1)
            tf.summary.image("Confusion Matrix", cm_image, step = epoch + 1)  # 将测试结果confuse matrix画到tensorboard
            validation_summary_writer.flush()
        # 训练完成保存模型
        print("Model saved at Epoch %d end ." % (epoch + 1,))
        model.save(save_path.format((epoch + 1), acc))
        val_epoch_acc.reset_states()
        val_epoch_loss.reset_states()
        #测试完成
        validation_summary_writer.flush()

        history = History(train_epoch_acc = train_acc, train_epoch_loss = train_loss, val_epoch_acc = val_acc,
                          val_epoch_loss = val_loss)
        plot_acc_loss(history = history, log_dir = log_dir)
    train_summary_writer.flush()
    validation_summary_writer.flush()
    train_summary_writer.close()
    validation_summary_writer.close()



if __name__ == "__main__":
    main()
