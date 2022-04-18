#!/bin/bash
python generate.py \
--ckpt /scratch/arturao/GANSketching22/checkpoint/horse_riders_original_sup30-10000-s90-fixD/0_net_G.pth \
--save_dir /scratch/arturao/GANSketching22/output/horse_riders_original_sup30-10000-s90-fixD/ \
--samples 2500 \
--use_supermask \
--sparsity 0.9 \