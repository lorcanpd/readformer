#!/bin/bash



LOG_DIR="logs/pretrain"

mkdir -p ${LOG_DIR}

READFORMER_DIR="/lustre/scratch126/casm/team274sb/lp23/readformer/readformer"
WANDB_API_KEY_PATH="/lustre/scratch126/casm/team274sb/lp23/.wandb_api"
SIF="/nfs/users/nfs_l/lp23/sifs/readformer.sif"
DATA_DIR="/lustre/scratch126/casm/team274sb/lp23/readformer/data/pretrain_bams"
METADATA_PATH="/lustre/scratch126/casm/team274sb/lp23/readformer/data/one_sample_metadata.csv"
MODEL_DIR="/lustre/scratch126/casm/team274sb/lp23/readformer/models"
GPU_MEMORY=40960
MEMORY=32768
CORES=4
NUM_ORDER=4
NUM_LAYERS=2
MIN_READ_QUALITY=20
BATCH_SIZE=128
EMB_DIM=128
MAX_SEQUENCE_LENGTH=8192
WARM_UP_EPOCHS=2
EPOCHS_AT_INTERVAL=1
ITERS_IN_EPOCH=50
CORRUPTION_RATE="variable"
PROPORTION_RANDOM=0.1
MAIN_LR=1e-3


#CORRUPTION_SCALE=0.5
NAME="TEST"

BASENAME="6hour_PROFILING_readformer_bs${BATCH_SIZE}_${EMB_DIM}d_${NUM_ORDER}g_${NUM_LAYERS}l"

#SCALES=( 0.5 0.75 0.9 )

SCALES=( 0.9 )

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
#BSUB -gpu "num=1:mode=exclusive_process:j_exclusive=yes:block=yes:gmem=${GPU_MEMORY}"
#BSUB -R 'span[hosts=1] span[ptile=${CORES}]'  # Allocate 4 CPU cores per node
#BSUB -R "select[mem>${MEMORY}] rusage[mem=${MEMORY}]" # span[hosts=1]"
#BSUB -W 6:00

module load cellgen/singularity

singularity exec --nv \
  --env LSB_DJOB_NUMPROC=${CORES} \
  --bind ${READFORMER_DIR}:/scripts/readformer \
  --bind ${DATA_DIR}:/data/pretrain/BAM \
  --bind ${METADATA_PATH}:/data/pretrain_metadata.csv \
  --bind ${MODEL_DIR}:/models \
  --bind ${WANDB_API_KEY_PATH}:/home/wandb_api_key \
  --pwd /scripts/readformer \
  ${SIF} \
  python3 /scripts/readformer/mlm_pretraining.py \
    --readformer \
    --metadata_path /data/pretrain_metadata.csv \
    --data_dir /data/pretrain/BAM \
    --wandb_api_path /home/wandb_api_key \
    --model_dir /models \
    --n_order ${NUM_ORDER} \
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
    --corruption_scale ${CORRUPTION_SCALE} \
    --name ${NAME} \
    --wandb \
    --profiling

EOF
  )

  if [[ $? -ne 0 ]]; then
    echo "Error submitting job with corruption rate scaling ${scale}"
    exit 1
  fi

  echo "Submitted job ${job_id} with corruption rate scaling ${scale}"
done
