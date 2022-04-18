#!/bin/bash
python train.py \
--name standing_cats_original4-75000-full \
--dataroot_sketch /scratch/arturao/GANSketching_old/data/sketch/photosketch/standing_cat4 \
--dataroot_image /scratch/arturao/GANSketching_old/data/image/cat_new --l_image 0.7 \
--disable_eval \
--g_pretrained /scratch/arturao/GANSketching22/pretrained/stylegan2-cat/netG.pth \
--d_pretrained /scratch/arturao/GANSketching22/pretrained/stylegan2-cat/netD.pth \
--diffaug_policy translation \
--no_wandb \
--max_iter 75001 \