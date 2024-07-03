#!/bin/bash

BASENAME="hyena_128dim_2group_2layer"

LOG_DIR="logs/pretrain"

READFORMER_DIR="/lustre/scratch126/casm/team274sb/lp23/readformer/readformer"
WANDB_API_KEY_PATH="/lustre/scratch126/casm/team274sb/lp23/.wandb_api"
SIF="/nfs/users/nfs_l/lp23/sifs/readformer.sif"
DATA_DIR="/lustre/scratch126/casm/team274sb/lp23/readformer/data/pretrain_symlinks"
METADATA_PATH="/lustre/scratch126/casm/team274sb/lp23/readformer/data/pretrain_metadata.csv"
MODEL_DIR="/lustre/scratch126/casm/team274sb/lp23/readformer/models"
GPU_MEMORY=16384
MEMORY=51200
CORES=4
NUM_HEADS=2
NUM_LAYERS=2
MIN_READ_QUALITY=20
BATCH_SIZE=64
EMB_DIM=128
MAX_SEQUENCE_LENGTH=8192
WARM_UP_EPOCHS=10
EPOCHS_AT_INTERVAL=2
ITERS_IN_EPOCH=1000
CORRUPTION_RATE="variable"
PROPORTION_RANDOM=0.1
MAIN_LR=1e-3
WANDB=true

SCALES=( 0.5 0.75 0.9 )

for scale in "${SCALES[@]}"; do
  # Set the arguments
  NAME="${BASENAME}_corrupt_${scale}"
  CORRUPTION_SCALE=${scale}

  job_id=$(bsub << EOF | grep -oE "[0-9]+"
#!/bin/bash
#BSUB -J ${NAME}
#BSUB -q gpu-normal
#BSUB -o ${LOG_DIR}/${NAME}_%J.out
#BSUB -e ${LOG_DIR}/${NAME}_%J.err
#BSUB -M ${MEMORY}
#BSUB -n ${CORES}
#BSUB -gpu "num=1:gmem=${GPU_MEMORY}"
#BSUB -R "select[mem>${MEMORY}] rusage[mem=${MEMORY}, ngpus_physical=1, gmem=${GPU_MEMORY}] span[hosts=1]"

module load cellgen/singularity

singularity exec --nv ${SIF} \
  --bind ${READFORMER_DIR}:/scripts/readformer \
  --bind ${DATA_DIR}:/data/pretrain_symlinks \
  --bind ${METADATA_PATH}:/data/pretrain_metadata.csv \
  --bind ${MODEL_DIR}:/models \
  --bind ${WANDB_API_KEY_PATH}:/home/wandb_api_key \
  python3 /scripts/readformer/mlm_pretraining.py \
    --metadata_path /data/pretrain_metadata.csv \
    --data_dir /data/pretrain_symlinks \
    --wandb_api_path /home/wandb_api_key \
    --model_dir /models \
    --num_heads ${NUM_HEADS} \
    --num_layers ${NUM_LAYERS} \
    --min_read_quality ${MIN_READ_QUALITY} \
    --batch_size ${BATCH_SIZE} \
    --emb_dim ${EMB_DIM} \
    --max_sequence_length ${MAX_SEQUENCE_LENGTH} \
    --warm_up_epochs ${WARM_UP_EPOCHS} \
    --epochs_at_interval ${EPOCHS_AT_INTERVAL} \
    --iters_in_epoch ${ITERS_IN_EPOCH} \
    --corruption_rate ${CORRUPTION_RATE} \
    --proportion_random ${PROPORTION_RANDOM} \
    --main_lr ${MAIN_LR} \
    --wandb ${WANDB} \
    --hyena \
    --corruption_scale ${CORRUPTION_SCALE} \
    --name ${NAME}

EOF
  )

  echo "Submitted job ${job_id} with corruption rate scaling ${scale}"
done
