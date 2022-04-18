#!/bin/bash
python train.py \
--name gabled_churches_augment30-75000-full \
--dataroot_sketch /scratch/arturao/GANSketching22/data/sketch/photosketch/gch \
--dataroot_image /scratch/arturao/GANSketching_old/data/image/church --l_image 0.7 \
--eval_dir /scratch/arturao/GANSketching_old/data/eval/gabled_church \
--g_pretrained /scratch/arturao/GANSketching_old/pretrained/stylegan2-church/netG.pth \
--d_pretrained /scratch/arturao/GANSketching_old/pretrained/stylegan2-church/netD.pth \
--diffaug_policy translation \
--no_wandb \
--max_iter 75001 \