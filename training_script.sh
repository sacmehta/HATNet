#!/usr/bin/env bash

#Base feature extractor
cnn_model='espnetv2'
cnn_model_scale='2.0'
cnn_model_weights='model/pretrained_cnn_models/espnetv2_s_'$cnn_model_scale'_imagenet_224x224.pth'


#Location of the data
img_dir='./wsi_dataset'
# WSI extendsion (without dot)
img_extn='tiff' # do not add .
# dataset name.
dataset='bbwsi' # breast biopsy wsi

# training and validation file splits
train_fname='./wsi_dataset/splits/train.txt'
val_fname='./wsi_dataset/splits/val.txt'

# bag and word size details
bag_size=1792
word_size=256


# Holistic attention model details
# head dimension in multi-head attention
head_dim=64 # We use 64. The same in Transformer paper (Attention is all you need)
attn_type='l2' # Function \Psi in the paper

# dropout rate
dropout_rate=0.4
# Project the CNN features to this dimensionality. Make sure that it is divisible by head_dimension
out_features=256 # head dimesnion is 64
attn_heads=$(($out_features/$head_dim))

# learning rate scheduler details
scheduler='multistep' # which scheduler
lr=1e-4 # learning rate
lr_decay=0.5 # decrease LR by this factor.
step_size=51 # At this epoch, we decrease the LR by 0.5.

epochs=100 # Number of training epochs
batch_size=1 # number of slides in a batch. In our experiments, we use 1. Otherwise OOM errors because of large image sizes
accum_count=8 # accumulate gradients for 8 steps before updating the weights. This allows us to use larger effective batch size
data_workers=4 # Data workers
optimizer='adam' # optimizer
warm_up_steps=600 # warm-up iterations
max_bsz_cnn_gpu0=100 #maximum words on GPU0. We will scale it by num_available GPUs in the code
loss_fn='ce' # loss function (ce stands for cross entropy)


# run the training code on two GPUs (GPU-0 and GPU-1)
CUDA_VISIBLE_DEVICES=0,1 python main_training.py --base-extractor $cnn_model --s $cnn_model_scale --weights $cnn_model_weights \
--img-dir $img_dir --img-extn $img_extn --dataset $dataset \
--train-file $train_fname --val-file $val_fname \
--bag-size $bag_size --word-size $word_size \
--scheduler $scheduler --lr $lr --lr-decay $lr_decay --step-size $step_size --epochs $epochs --batch-size $batch_size \
--accum-count $accum_count --optim $optimizer \
--log-interval 5 --data-workers $data_workers --warm-up --warm-up-iterations $warm_up_steps \
--attn-heads $attn_heads --dropout $dropout_rate --out-features $out_features --class-weights \
--max-bsz-cnn-gpu0 $max_bsz_cnn_gpu0 --attn-type $attn_type --loss-fn $loss_fn
