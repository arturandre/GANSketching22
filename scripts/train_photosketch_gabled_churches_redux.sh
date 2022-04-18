#!/bin/bash
python train.py \
--name gabled_churches_original30-10000 \
--dataroot_sketch /scratch/arturao/GANSketching_old/data/sketch/photosketch/gabled_church \
--dataroot_image /scratch/arturao/GANSketching_old/data/image/church --l_image 0.7 \
--disable_eval \
--g_pretrained /scratch/arturao/GANSketching_old/pretrained/stylegan2-church/netG.pth \
--d_pretrained /scratch/arturao/GANSketching_old/pretrained/stylegan2-church/netD.pth \
--diffaug_policy translation \
--no_wandb \
--max_iter 75001 \
--resume_iter 10000 \