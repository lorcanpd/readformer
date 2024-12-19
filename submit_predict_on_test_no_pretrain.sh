#!/bin/bash


# Fixed params and dir
READFORMER_DIR="/lustre/scratch126/casm/team274sb/lp23/readformer/readformer"
WANDB_API_KEY_PATH="/lustre/scratch126/casm/team274sb/lp23/.wandb_api"
SIF="/nfs/users/nfs_l/lp23/sifs/readformer.sif"
DATA_DIR="/lustre/scratch126/casm/team274sb/lp23/readformer/data/finetune"
PREDICTION_DIR="${DATA_DIR}/predictions"
METADATA_DIR="${DATA_DIR}/folds"
BAM_DIR="${DATA_DIR}/BAM"
MUT_BAM="${BAM_DIR}/mutation_reads.bam"
MUT_BAI="${MUT_BAM}.bai"

ART_BAM="${BAM_DIR}/HG002_artefacts.bam"
ART_BAI="${ART_BAM}.bai"

MODEL_DIR="/lustre/scratch126/casm/team274sb/lp23/readformer/models"

#PRETRAIN_DIR="${MODEL_DIR}/pretrained"

CORES=12
GPU_MEMORY=40000
MEMORY=8192

BATCH_SIZE=200
KERNEL_SIZE=7
NUM_ORDER=2
PROJECT="no_pretrain"
FOLD=0
LOG_DIR="logs/finetune/${PROJECT}/fold_${FOLD}"
mkdir -p ${LOG_DIR}



#EMB_DIMS=( 512 256  256 )
#HEAD_NUMS=( 32  16   16 )
#LAYER_NUMS=( 4   1    4 )
#NUM_HYENAS=( 5   0    5 )
#NUM_ATTENS=( 1  24    1 )

#EMB_DIMS=( 512 256 )
#HEAD_NUMS=( 32  16 )
#LAYER_NUMS=( 4   4 )
#NUM_HYENAS=( 5   5 )
#NUM_ATTENS=( 1   1 )

EMB_DIMS=( 256 )
HEAD_NUMS=( 16 )
LAYER_NUMS=( 1 )
NUM_HYENAS=( 0 )
NUM_ATTENS=( 24 )


for i in "${!EMB_DIMS[@]}"; do
  EMB_DIM=${EMB_DIMS[$i]}
  NUM_HEADS=${HEAD_NUMS[$i]}
  NUM_LAYERS=${LAYER_NUMS[$i]}
  NUM_HYENA=${NUM_HYENAS[$i]}
  NUM_ATTENTION=${NUM_ATTENS[$i]}

  MODEL_NAME="${EMB_DIM}d_${NUM_LAYERS}l_${NUM_HYENA}h_${NUM_ATTENTION}a_${NUM_HEADS}h"
  FINETUNE_SAVE_DIR="${MODEL_DIR}/finetune/${PROJECT}/${MODEL_NAME}/fold_${FOLD}"

  VALIDATION_OUTPUT_DIR="${PREDICTION_DIR}/${PROJECT}/${MODEL_NAME}"
  mkdir -p "${VALIDATION_OUTPUT_DIR}"

  job_id=$(bsub << EOF | grep -oE "[0-9]+"
#!/bin/bash
#BSUB -J ${MODEL_NAME}
#BSUB -q gpu-basement
#BSUB -m "farm22-gpu0203"
#BSUB -o ${LOG_DIR}/${MODEL_NAME}_%J.out
#BSUB -e ${LOG_DIR}/${MODEL_NAME}_%J.err
#BSUB -M ${MEMORY}
#BSUB -n ${CORES}
#BSUB -gpu "num=1:mode=exclusive_process:j_exclusive=yes:block=yes:gmem=${GPU_MEMORY}"
#BSUB -R 'span[hosts=1] span[ptile=${CORES}]'
#BSUB -R "select[mem>${MEMORY}] rusage[mem=${MEMORY}]" # span[hosts=1]"

export CUDA_LAUNCH_BLOCKING=1

module load cellgen/singularity

singularity exec --nv \
  --env LSB_DJOB_NUMPROC=${CORES} \
  --bind ${READFORMER_DIR}:/scripts/readformer \
  --bind ${MUT_BAM}:/data/pretrain/BAM/mutation_reads.bam \
  --bind ${MUT_BAI}:/data/pretrain/BAM/mutation_reads.bam.bai \
  --bind ${ART_BAM}:/data/pretrain/BAM/HG002_artefacts.bam \
  --bind ${ART_BAI}:/data/pretrain/BAM/HG002_artefacts.bam.bai \
  --bind ${METADATA_DIR}/train_fold_${FOLD}.csv:/data/pretrain/VCF/train_fold_${FOLD}.csv \
  --bind ${METADATA_DIR}/test_fold_${FOLD}.csv:/data/pretrain/VCF/test_fold_${FOLD}.csv \
  --bind ${FINETUNE_SAVE_DIR}:/models \
  --bind ${VALIDATION_OUTPUT_DIR}:/nst_dir \
  --bind ${WANDB_API_KEY_PATH}:/home/wandb_api_key \
  --pwd /scripts/readformer \
  ${SIF} \
  python3 /scripts/readformer/finetune.py \
    --project ${PROJECT} \
    --name ${MODEL_NAME} \
    --emb_dim ${EMB_DIM} \
    --num_heads ${NUM_HEADS} \
    --num_layers ${NUM_LAYERS} \
    --n_order ${NUM_ORDER} \
    --kernel_size ${KERNEL_SIZE} \
    --num_hyena ${NUM_HYENA} \
    --num_attention ${NUM_ATTENTION} \
    --batch_size ${BATCH_SIZE} \
    --readformer \
    --finetune_save_dir /models \
    --finetune_metadata_dir /data/pretrain/VCF \
    --mutation_bam_path /data/pretrain/BAM/mutation_reads.bam \
    --artefact_bam_path /data/pretrain/BAM/HG002_artefacts.bam \
    --fold ${FOLD} \
    --validation_output_dir /nst_dir

EOF
  )
  if [[ -z "${job_id}" ]]; then
    echo "Job submission failed for ${MODEL_NAME} using fold ${FOLD}"
    exit 1
  fi

  echo "Submitted job ${job_id} for ${MODEL_NAME} using fold ${FOLD}"

done

