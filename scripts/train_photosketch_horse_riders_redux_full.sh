#!/bin/bash
python train.py \
--name horse_riders_original30-75000-full \
--dataroot_sketch /scratch/arturao/GANSketching_old/data/sketch/photosketch/horse_riders \
--dataroot_image /scratch/arturao/GANSketching22/data/image/horse2 --l_image 0.7 \
--disable_eval \
--g_pretrained /scratch/arturao/GANSketching_old/pretrained/stylegan2-horse/netG.pth \
--d_pretrained /scratch/arturao/GANSketching_old/pretrained/stylegan2-horse/netD.pth \
--diffaug_policy translation \
--no_wandb \
--max_iter 75001 \