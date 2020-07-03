#!/usr/bin/env bash

img_dir='./wsi_dataset' # location of the images folder
weights_file='model_zoo/espnetv2_weights_bag_1792_word_256_softmax_l2.pth' # location of the weights file
config_file='model_zoo/espnetv2_config_bag_1792_word_256_softmax_l2.json' # path to the config file. Should be in the same folder where model weights are saved
test_file='./wsi_dataset/splits/test.txt' # name of the file that contains image and class labels

# run the evaluation code
CUDA_VISIBLE_DEVICES=0 python main_evaluation.py --img-dir $img_dir --weights-test $weights_file --config-file $config_file --test-file $4