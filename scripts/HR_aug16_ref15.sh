#!/bin/bash
python train.py \
--name HR_augs16-ref15-75000-full \
--dataroot_sketch /scratch/arturao/GANSketching22/data/sketch/photosketch/HR_aug16_ref15 \
--dataroot_image /scratch/arturao/GANSketching22/data/image/horse2 --l_image 0.7 \
--disable_eval \
--g_pretrained /scratch/arturao/GANSketching22/pretrained/stylegan2-horse/netG.pth \
--d_pretrained /scratch/arturao/GANSketching22/pretrained/stylegan2-horse/netD.pth \
--diffaug_policy translation \
--no_wandb \
--max_iter 75001 \