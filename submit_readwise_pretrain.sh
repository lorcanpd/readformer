#!/bin/bash



LOG_DIR="logs/pretrain/readwise_only"

mkdir -p ${LOG_DIR}

READFORMER_DIR="/lustre/scratch126/casm/team274sb/lp23/readformer/readformer"
WANDB_API_KEY_PATH="/lustre/scratch126/casm/team274sb/lp23/.wandb_api"
SIF="/nfs/users/nfs_l/lp23/sifs/readformer.sif"
DATA_DIR="/lustre/scratch126/casm/team274sb/lp23/readformer/data/pretrain_bams"
METADATA_PATH="/lustre/scratch126/casm/team274sb/lp23/readformer/data/one_sample_metadata.csv"
MODEL_DIR="/lustre/scratch126/casm/team274sb/lp23/readformer/models/read_only_pretrain"
GPU_MEMORY=80000
MEMORY=32768
MAX_ITERS=1000
CORES=12
NUM_ORDER=2
NUM_HEADS=8
KERNEL_SIZE=3
NUM_LAYERS=1
MIN_READ_QUALITY=10
BATCH_SIZE=2048
EMB_DIM=64
MAX_SEQUENCE_LENGTH=256  # Single reads
WARM_UP_EPOCHS=2
#EPOCHS_AT_INTERVAL=1
ITERS_IN_EPOCH=2000
MAX_ITERS=2000
CORRUPTION_RATE=0.2
PROPORTION_RANDOM=0.25
MIXING_ALPHA=0.4
MAIN_LR=0.0032

HYENA_NUMS=( 4 )

NAME="_hyena_only_pretrain"

for NUM_HYENA in "${HYENA_NUMS[@]}"; do

  job_id=$(bsub << EOF | grep -oE "[0-9]+"
#!/bin/bash
#BSUB -J ${NAME}
#BSUB -q gpu-basement
#BSUB -o ${LOG_DIR}/${NAME}_%J.out
#BSUB -e ${LOG_DIR}/${NAME}_%J.err
#BSUB -M ${MEMORY}
#BSUB -n ${CORES}
#BSUB -gpu "num=1:mode=exclusive_process:j_exclusive=yes:block=yes:gmem=${GPU_MEMORY}"
#BSUB -R 'span[hosts=1] span[ptile=${CORES}]'  # Allocate 4 CPU cores per node
#BSUB -R "select[mem>${MEMORY}] rusage[mem=${MEMORY}]" # span[hosts=1]"

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
  python3 /scripts/readformer/pretrain_readwise_only.py \
    --readformer \
    --metadata_path /data/pretrain_metadata.csv \
    --data_dir /data/pretrain/BAM \
    --wandb_api_path /home/wandb_api_key \
    --model_dir /models \
    --n_order ${NUM_ORDER} \
    --kernel_size ${KERNEL_SIZE} \
    --num_layers ${NUM_LAYERS} \
    --num_heads ${NUM_HEADS} \
    --num_hyena ${NUM_HYENA} \
    --min_read_quality ${MIN_READ_QUALITY} \
    --batch_size ${BATCH_SIZE} \
    --emb_dim ${EMB_DIM} \
    --max_sequence_length ${MAX_SEQUENCE_LENGTH} \
    --warm_up_epochs ${WARM_UP_EPOCHS} \
    --iters_in_epoch ${ITERS_IN_EPOCH} \
    --corruption_rate ${CORRUPTION_RATE} \
    --proportion_random ${PROPORTION_RANDOM} \
    --main_lr ${MAIN_LR} \
    --name ${NAME} \
    --max_iters ${MAX_ITERS} \
    --mixing_alpha ${MIXING_ALPHA} \
    --wandb

EOF
  )

  if [[ $? -ne 0 ]]; then
    echo "Error submitting readformer job with ${NUM_HYENA} stacked hyenas"
    exit 1
  fi

  echo "Submitted readformer job ${job_id} with ${NUM_HYENA} stacked hyenas"
done
