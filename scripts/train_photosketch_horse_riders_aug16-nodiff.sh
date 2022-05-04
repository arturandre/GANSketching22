#!/bin/bash
python train.py \
--name horse_riders_augs16-75000-full-nodiff \
--dataroot_sketch /scratch/arturao/GANSketching22/data/sketch/photosketch/horse_riders_augs16 \
--dataroot_image /scratch/arturao/GANSketching22/data/image/horse2 --l_image 0.7 \
--disable_eval \
--g_pretrained /scratch/arturao/GANSketching22/pretrained/stylegan2-horse/netG.pth \
--d_pretrained /scratch/arturao/GANSketching22/pretrained/stylegan2-horse/netD.pth \
--no_wandb \
--max_iter 75001 \