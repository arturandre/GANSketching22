#!/bin/bash
python train.py \
--name horse_riders_original_sup30-10000-s90 \
--dataroot_sketch /scratch/arturao/GANSketching_old/data/sketch/photosketch/horse_riders \
--dataroot_image /scratch/arturao/GANSketching_old/data/image/horse --l_image 0.7 \
--disable_eval \
--eval_dir /scratch/arturao/GANSketching_old/data/eval/horse_riders \
--g_pretrained /scratch/arturao/GANSketching_old/pretrained/stylegan2-horse/netG.pth \
--d_pretrained /scratch/arturao/GANSketching_old/pretrained/stylegan2-horse/netD.pth \
--diffaug_policy translation \
--no_wandb \
--max_iter 10001 \
--use_supermask \
--sparsity 0.9 \