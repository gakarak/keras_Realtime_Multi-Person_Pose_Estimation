#!/bin/bash


export CUDA_VISIBLE_DEVICES="1"

runpy="train_pose_mod2_mobnet.py"

python3 ${runpy}
