EXP_NAME=slow_decay_likelihood
NUM_GPUS=5
OFFSET=0

for i in $(seq $(($OFFSET + 1)) $(($OFFSET + $NUM_GPUS)))
do 
sbatch scripts/savio_run.sh ${EXP_NAME}${i}
done
