#!/bin/bash


pgpu='0.8'
pbatch='32'
palpha='0.5'
pstages='2'
pport='5555'

export CUDA_VISIBLE_DEVICES="0"

runpy="train_pose_mod3_mobnet3_v2.py"
##train_pose_mod3_mobnet.py"

python3 ${runpy} --batch=${pbatch} --gpumem=${pgpu} --palpha=${palpha} --pstages=${pstages} --port=${pport}
#--weights=${}
