#!/bin/bash
python train_hypernet.py \
--name horse_riders_original30-75000-full-hyper-1024 \
--dataroot_sketch /scratch/arturao/GANSketching22/data/sketch/photosketch/horse_riders \
--sketch_channel 3 \
--dataroot_image /scratch/arturao/GANSketching22/data/image/horse2 --l_image 0.7 \
--eval_dir /scratch/arturao/GANSketching_old/data/eval/horse_riders \
--g_pretrained /scratch/arturao/GANSketching22/pretrained/stylegan2-horse/netG.pth \
--d_pretrained /scratch/arturao/GANSketching22/pretrained/stylegan2-horse/netD.pth \
--diffaug_policy translation \
--no_wandb \
--max_iter 75001 \
--hypernet_params 1024 \