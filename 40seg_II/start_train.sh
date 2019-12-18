python train_ecg.py \
 --output_folder=/home/tianliang/out/model_seg40_II_250hz_4s_20191216_continue \
 --train_data_path=/deeplearn_data2/data_center/fragment_datas/Channel_II/NotContainNoise/Train/NonPureN \
 --validation_data_path=/deeplearn_data2/data_center/fragment_datas/Channel_II/NotContainNoise/Test/NonPureN  \
 --save_format=hdf5 \
 --which_gpu=3 \
 --batch_size=500 \
 --epochs=100 \
 --regularizer=5e-4 \
 --num_classes=12 \
 --initial_learning_rate=0.001 \
 --learning_rate_decay_factor=0.9 \
 --num_epochs_per_decay=1 \
 --continue_train=True \
 --pretrained_model_path=/home/tianliang/out/model_seg40_II_250hz_4s_20191216/hdf5_models_20191218_170100/ckpt_epoch01_val_acc0.57.hdf5 \
 --moving_average_decay_factor=0.99 \
# python train_ecg.py \
# --output_folder=/raid/data1/out/model_I_250hz_4s_20191210_2 \
# --train_data_path=/deeplearn_data2/experimental_data/191126/train_datas_I \
# --validation_data_path=/deeplearn_data2/experimental_data/191126/test_datas_I \
# --save_format=saved_model \
# --which_gpu=5 \
# --batch_size=256 \
# --epochs=2 \
# --regularizer=5e-4 \
# --lr_decay_epochs=1 \
# --num_classes=12 \
# --initial_learning_rate=0.002