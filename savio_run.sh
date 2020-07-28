#!/bin/bash
# Job name:
#SBATCH --job-name=smote2
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
for TRIAL in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29
do
EXP_NUM=new_smote${TRIAL}
mkdir logs/${EXP_NUM}
python -u test.py \
        --name ${EXP_NUM} \
        --batch_size 1024 \
        --epochs 100 \
        --smote True \
>& logs/${EXP_NUM}/train.log
done
