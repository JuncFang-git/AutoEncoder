cd .. &&
###
 # @Author: Juncfang
 # @Date: 2022-06-16 15:51:32
 # @LastEditTime: 2022-06-16 15:58:29
 # @LastEditors: Juncfang
 # @Description: 
 # @FilePath: /AutoEncoder/script/train-auto_encoder.sh
 #  
### 
CUDA_VISIBLE_DEVICES=0 \
python train.py \
--name auto_encoder-test \
--dataroot /home/junkai/dataset/ddy2/style1/combine/half \
--checkpoints_dir ./checkpoints/ \
--resize_or_crop none \
--ngf 16 \
--batchSize 1 \
--niter 100 \
--niter_decay 300 \
--save_latest_freq 20000 \
--save_epoch_freq 10 \
--lambda_feat 0.01 \
--use_l1_loss \
# --no_vgg_loss \ 
# --resize_or_crop scale_width_and_crop --loadSize 512 --fineSize 512 \
# --verbose \
# --debug \