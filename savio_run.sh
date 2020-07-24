#!/bin/bash
# Job name:
#SBATCH --job-name=thin_fc
#
# Account:
#SBATCH --account=fc_ntugame
#
# Partition:
#SBATCH --partition=savio2_1080ti
#
# QoS:
#SBATCH --qos=savio_normal
#
# Number of nodes:
#SBATCH --nodes=1
#
# Number of tasks (one for each GPU desired for use case) (example):
#SBATCH --ntasks=1
#
# Processors per task (please always specify the total number of processors twice the number of GPUs):
#SBATCH --cpus-per-task=2
#
#Number of GPUs, this can be in the format of "gpu:[1-4]", or "gpu:K80:[1-4] with the type included
#SBATCH --gres=gpu:1
#
# Wall clock limit:
#SBATCH --time=72:00:00
#
## Command(s) to run (example):
EXP_NUM=baseline4
mkdir logs/${EXP_NUM}
python -u test.py \
        --name ${EXP_NUM} \
        --batch_size 1024 \
        --lr 1e-4 \
        --hidden_layers 512 \
        --iterations 1 \
>& logs/${EXP_NUM}/train.log
