#!/bin/bash
python train.py \
--name horse_riders_original30-full-top4k \
--dataroot_sketch /scratch/arturao/GANSketching22/data/sketch/photosketch/horse_riders \
--dataroot_image /scratch/arturao/GANSketching22/data/image/horse2 --l_image 0.7 \
--eval_dir /scratch/arturao/GANSketching22/data/eval/horse_riders \
--g_pretrained /scratch/arturao/GANSketching22/pretrained/stylegan2-horse/netG.pth \
--d_pretrained /scratch/arturao/GANSketching22/pretrained/stylegan2-horse/netD.pth \
--diffaug_policy translation \
--no_wandb \
--fine_tune_top_k 4096 \
--max_iter 75001 \
