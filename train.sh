#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,3,4,5,6,7

#accelerate launch --config_file default_config.yaml  pretrain.py

#accelerate launch --gpu_ids='0,1' --num_processes=2 
python pretrain.py

