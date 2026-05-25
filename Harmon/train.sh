# export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH=./
export NCCL_NVLS_ENABLE=0

CONFIG_NAME=${1:-"UniMRG"}
CONFIG_FILE="configs/examples/${CONFIG_NAME}.py"

export LAUNCHER="torchrun \
    --nproc_per_node=${GPUS:-8} \
    --nnodes=${WORLD_SIZE:-1} \
    --node_rank=${RANK:-0} \
    --master_addr=${MASTER_ADDR:-"127.0.0.1"} \
    --master_port=${MASTER_PORT:-12348} \
    "

export CMD="scripts/train.py \
$CONFIG_FILE \
--launcher pytorch \
--deepspeed deepspeed_zero2"

echo $LAUNCHER
echo $CMD

bash -c "$LAUNCHER $CMD"

sleep 60s
