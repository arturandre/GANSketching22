#!/bin/bash
python train.py \
--name standing_cats_original30-10000 \
--dataroot_sketch /scratch/arturao/GANSketching_old/data/sketch/photosketch/standing_cat \
--dataroot_image /scratch/arturao/GANSketching_old/data/image/cat --l_image 0.7 \
--disable_eval \
--g_pretrained /scratch/arturao/GANSketching22/pretrained/stylegan2-cat/netG.pth \
--d_pretrained /scratch/arturao/GANSketching22/pretrained/stylegan2-cat/netD.pth \
--diffaug_policy translation \
--no_wandb \
--max_iter 75001 \
--resume_iter 67500 \