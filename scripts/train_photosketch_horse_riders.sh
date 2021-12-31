#!/bin/bash
python train.py \
--name horse_riders_augment \
--dataroot_sketch /scratch/arturao/GANSketching/data/sketch/photosketch/horse_riders \
--dataroot_image /scratch/arturao/GANSketching/data/image/horse --l_image 0.7 \
--eval_dir /scratch/arturao/GANSketching/data/eval/horse_riders \
--g_pretrained /scratch/arturao/GANSketching/pretrained/stylegan2-horse/netG.pth \
--d_pretrained /scratch/arturao/GANSketching/pretrained/stylegan2-horse/netD.pth \
--diffaug_policy translation \
--no_wandb \
--disable_eval \
--max_epoch 80