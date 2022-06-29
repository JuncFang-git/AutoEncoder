cd .. &&
###
 # @Author: Juncfang
 # @Date: 2022-06-16 15:51:32
 # @LastEditTime: 2022-06-29 10:58:19
 # @LastEditors: Juncfang
 # @Description: 
 # @FilePath: /AutoEncoder/script/train-auto_encoder.sh
 #  
### 
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python train.py \
--name auto_encoder-3945 \
--dataroot /home/junkai/dataset/align-half-6-28 \
--checkpoints_dir ./checkpoints/ \
--resize_or_crop none \
--ngf 16 \
--batchSize 1 \
--niter 100 \
--niter_decay 300 \
--save_latest_freq 20000 \
--save_epoch_freq 10 \
--lambda_feat 1 \
--use_l1_loss \

# --debug \
# --no_vgg_loss \ 
# --resize_or_crop scale_width_and_crop --loadSize 512 --fineSize 512 \
# --verbose \