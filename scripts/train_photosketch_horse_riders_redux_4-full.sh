#!/bin/bash
python train.py \
--name horse_riders_original4-75000-full \
--dataroot_sketch /scratch/arturao/GANSketching_old/data/sketch/photosketch/horse_riders4 \
--dataroot_image /scratch/arturao/GANSketching_old/data/image/horse2 --l_image 0.7 \
--eval_dir /scratch/arturao/GANSketching_old/data/eval/horse_riders \
--g_pretrained /scratch/arturao/GANSketching_old/pretrained/stylegan2-horse/netG.pth \
--d_pretrained /scratch/arturao/GANSketching_old/pretrained/stylegan2-horse/netD.pth \
--diffaug_policy translation \
--no_wandb \
--max_iter 75001 \
--batch 4\