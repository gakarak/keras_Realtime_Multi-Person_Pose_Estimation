#!/bin/bash


pgpu='0.8'
pbatch='16'
palpha='0.5'
pstages='6'
pport='6557'

fweights='../../weights_mobilenet_best_a0.5_s4.h5'

export CUDA_VISIBLE_DEVICES="1"

runpy="train_pose_mod3_mobnet.py"

python3 ${runpy} --batch=${pbatch} --gpumem=${pgpu} --palpha=${palpha} --pstages=${pstages} --port=${pport} \
    --weights=${fweights}
