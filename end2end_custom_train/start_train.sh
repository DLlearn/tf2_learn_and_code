python  train_ecg.py \
 --output_folder=/raid/data1/out/model_II_250hz_4s_20191213 \
 --train_data_path=/deeplearn_data2/experimental_data/191210/Channel_II/NotContainNoise/NonPureN/Train \
 --validation_data_path=/deeplearn_data2/experimental_data/191210/Channel_II/NotContainNoise/NonPureN/Test \
 --save_format=hdf5 \
 --which_gpu=1 \
 --batch_size=125 \
 --epochs=2 \
 --regularizer=5e-4 \
 --lr_decay_epochs=1 \
 --num_classes=12 \
 --initial_learning_rate=0.002
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