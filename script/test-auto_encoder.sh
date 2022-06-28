
###
 # @Author: Juncfang
 # @Date: 2022-06-17 13:53:21
 # @LastEditTime: 2022-06-17 14:42:56
 # @LastEditors: Juncfang
 # @Description: 
 # @FilePath: /AutoEncoder/script/test-auto_encoder.sh
 #  
### 
cd .. &&
CUDA_VISIBLE_DEVICES=1 \
python test.py \
--which_epoch 400 \
--name auto_encoder-test \
--dataroot /home/junkai/dataset/ddy2/style1/combine/half \
--results_dir ./test_result/ \
--resize_or_crop none \
--ngf 16 \
--batchSize 1 \