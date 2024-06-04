#!/bin/bash

port=$(python get_free_port.py)
GPU=1

alias exp="python -m torch.distributed.launch --master_port=${port} --nproc_per_node=${GPU} tools/train_incremental.py"
shopt -s expand_aliases

# FIRST STEP
python -m torch.distributed.launch
--master_port
1995
--nproc_per_node=1
ADT-ILDFD/tools/train_first_step.py
-c
ADT-ILDFD/configs/neu_cfg/3-3/e2e_faster_rcnn_R_50_C4_4x.yaml

#GET OLD TEACHER
python ADT-ILDFD/tools/trim_detectron_model.py


#GET NEW TEACHER
#find 'def train(cfg, local_rank, distributed)' in 'train_first_step.py'
#    update 'extra_checkpoint_data = checkpointer.load(cfg.MODEL.WEIGHT,load_CTN=False)'
#to'extra_checkpoint_data = checkpointer.load(cfg.MODEL.WEIGHT,load_CTN=True)'
#
#find' def __init__（）：' in 'ADT-ILDFD\maskrcnn_benchmark\data\datasets\defect.py'
#update        'self.CTN = False'
#to       'self.CTN = True'
# update ADT-ILDFD/configs/neu_cfg/3-3/e2e_faster_rcnn_R_50_C4_4x.yaml ,then run FIRST STEP


# INCREMENTAL STEPS
python -m torch.distributed.launch
--master_port
1995
--nproc_per_node=1
ADT-ILDFD/tools/train_incremental_2teacher.py
-t
3-3
-n
ADT-ILDFD
--rpn
--uce
--feat
std
--dist_type
uce
--cls
0.1
--CTN
0.1
--RCN
5
