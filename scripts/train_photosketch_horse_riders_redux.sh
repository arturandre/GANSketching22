#!/bin/bash
python train.py \
--name horse_riders_original30-10000 \
--dataroot_sketch /scratch/arturao/GANSketching_old/data/sketch/photosketch/horse_riders \
--dataroot_image /scratch/arturao/GANSketching_old/data/image/horse --l_image 0.7 \
--disable_eval \
--eval_dir /scratch/arturao/GANSketching_old/data/eval/horse_riders \
--g_pretrained /scratch/arturao/GANSketching22/checkpoint/horse_riders_original30-10000/7500_net_G.pth \
--d_pretrained /scratch/arturao/GANSketching22/checkpoint/horse_riders_original30-10000/7500_net_D_image.pth \
--diffaug_policy translation \
--no_wandb \
--max_iter 10000 \
--resume_iter 7500 \