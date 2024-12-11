#!/bin/bash

LOG_DIR="logs/readwise_pretrain/"

mkdir -p ${LOG_DIR}

READFORMER_DIR="/lustre/scratch126/casm/team274sb/lp23/readformer/readformer"
WANDB_API_KEY_PATH="/lustre/scratch126/casm/team274sb/lp23/.wandb_api"
SIF="/nfs/users/nfs_l/lp23/sifs/readformer.sif"
DATA_DIR="/lustre/scratch126/casm/team274sb/lp23/readformer/data/pretrain_bams"
METADATA_PATH="/lustre/scratch126/casm/team274sb/lp23/readformer/data/pretrain_subsample_metadata.csv"
MODEL_DIR="/lustre/scratch126/casm/team274sb/lp23/readformer/models/read_only_pretrain"
VAL_BATCH_DIR="/lustre/scratch126/casm/team274sb/lp23/readformer/data/validation_batch"
GPU_MEMORY=61440
MEMORY=8192
MAX_ITERS=300000
CORES=12
#NUM_HYENA=3
NUM_ORDER=2
#NUM_HEADS=8
KERNEL_SIZE=7
#NUM_LAYERS=1
MIN_READ_QUALITY=30
BATCH_SIZE=128
#EMB_DIM=64
MAX_SEQUENCE_LENGTH=100  # Single reads
#EPOCHS_AT_INTERVAL=1
ITERS_IN_EPOCH=10000
CORRUPTION_RATE=0.15
PROPORTION_RANDOM=0.1
MAIN_LR=3e-4


NAME="pretrain"

PROJECT="pretrain"


EMB_DIMS=( 512 256 ) #512 )
HEAD_NUMS=( 32  16 ) # 32 )
LAYER_NUMS=( 4   4 ) #  1 )
NUM_HYENAS=( 5   0 ) # 24 )
NUM_ATTENS=( 1  6 ) #  0 )


for i in "${!EMB_DIMS[@]}"; do
  EMB_DIM=${EMB_DIMS[$i]}
  NUM_HEADS=${HEAD_NUMS[$i]}
  NUM_LAYERS=${LAYER_NUMS[$i]}
  NUM_HYENA=${NUM_HYENAS[$i]}
  NUM_ATTENTION=${NUM_ATTENS[$i]}
  JOBNAME="${NAME}_${EMB_DIM}d_${NUM_LAYERS}l_${NUM_HYENA}h_${NUM_ATTENTION}a_${NUM_HEADS}h"
  job_id=$(bsub << EOF | grep -oE "[0-9]+"
#!/bin/bash
#BSUB -J ${JOBNAME}
#BSUB -q gpu-basement
#BSUB -m "farm22-gpu0203"
#BSUB -o ${LOG_DIR}/${JOBNAME}_%J.out
#BSUB -e ${LOG_DIR}/${JOBNAME}_%J.err
#BSUB -M ${MEMORY}
#BSUB -n ${CORES}
#BSUB -gpu "num=1:mode=exclusive_process:j_exclusive=yes:block=yes:gmem=${GPU_MEMORY}"
#BSUB -R 'span[hosts=1] span[ptile=${CORES}]'
#BSUB -R "select[mem>${MEMORY}] rusage[mem=${MEMORY}]" # span[hosts=1]"

module load cellgen/singularity

singularity exec --nv \
  --env LSB_DJOB_NUMPROC=${CORES} \
  --bind ${READFORMER_DIR}:/scripts/readformer \
  --bind ${DATA_DIR}:/data/pretrain/BAM \
  --bind ${METADATA_PATH}:/data/pretrain_metadata.csv \
  --bind ${MODEL_DIR}:/models \
  --bind ${WANDB_API_KEY_PATH}:/home/wandb_api_key \
  --bind ${VAL_BATCH_DIR}:/nst_dir \
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
    --num_attention ${NUM_ATTENTION} \
    --min_read_quality ${MIN_READ_QUALITY} \
    --batch_size ${BATCH_SIZE} \
    --emb_dim ${EMB_DIM} \
    --max_sequence_length ${MAX_SEQUENCE_LENGTH} \
    --iters_in_epoch ${ITERS_IN_EPOCH} \
    --corruption_rate ${CORRUPTION_RATE} \
    --proportion_random ${PROPORTION_RANDOM} \
    --main_lr ${MAIN_LR} \
    --name ${NAME} \
    --project ${PROJECT} \
    --max_iters ${MAX_ITERS} \
    --validation_dir /nst_dir \
    --adam \
    --wandb
#    --load_latest_checkpoint True

EOF
    )

    if [[ $? -ne 0 ]]; then
      echo "Error submitting readformer job with ${NUM_LAYERS} layers, ${NUM_HYENA} hyenas, ${NUM_ATTENTION} attention layers with ${EMB_DIM} emb dim and ${NUM_HEADS} heads"
      exit 1
    fi

    echo "Submitted readformer job ${job_id} with ${NUM_LAYERS} layers, ${NUM_HYENA} hyenas, ${NUM_ATTENTION} attention layers with ${EMB_DIM} emb dim and ${NUM_HEADS} heads"
#  done
done
