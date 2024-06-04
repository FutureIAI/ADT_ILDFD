# Adaptive dual teacher incremental learning for defect detection of flexible printed circuit (https://www.sciencedirect.com/science/article/pii/S0045790624002659?dgcid=author)

## Official PyTorch implementatation based on Detetron (v1) - Fabio Cermelli, Antonino Geraci, Dario Fontanel, Barbara Caputo

# How to run
## Install
Please, follow the instruction provided by Detectron 1 and found in install.md

## Dataset
NEU-DET（http://faculty.neu.edu.cn/songkechen/zh_CN/zdylm/263270/list/）
## Run!
We provide scripts to run the experiments in the paper (ADT-ILDFD).
You can find one scripts: `run.sh`. The file can be used to run, respectively: single-step detection settings (3-3).

You can play with the following parameters to obtain all the results in the paper:
- `--CTN` is a float indicating the weight of the new teacher model;
- `--RCN` is a float indicating the weight of the Faster RCNN;

# Note
The code is partially based on MMA(https://github.com/fcdl94/mma)
