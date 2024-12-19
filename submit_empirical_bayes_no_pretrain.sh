#!/bin/bash

# Fixed params and dir
READFORMER_DIR="/lustre/scratch126/casm/team274sb/lp23/readformer/readformer"
SIF="/nfs/users/nfs_l/lp23/sifs/readformer.sif"
DATA_DIR="/lustre/scratch126/casm/team274sb/lp23/readformer/data/finetune"
PREDICTION_DIR="${DATA_DIR}/predictions"

CORES=4
MEMORY=16384

PROJECT="no_pretrain"
FOLD=0
LOG_DIR="logs/empirical_bayes/${PROJECT}/fold_${FOLD}"
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

  VALIDATION_OUTPUT_DIR="${PREDICTION_DIR}/${PROJECT}/${MODEL_NAME}"
  mkdir -p "${VALIDATION_OUTPUT_DIR}"

  job_id=$(bsub << EOF | grep -oE "[0-9]+"
#!/bin/bash
#BSUB -J ${MODEL_NAME}
#BSUB -q normal
#BSUB -o ${LOG_DIR}/${MODEL_NAME}_%J.out
#BSUB -e ${LOG_DIR}/${MODEL_NAME}_%J.err
#BSUB -M ${MEMORY}
#BSUB -n ${CORES}
#BSUB -R "select[mem>${MEMORY}] rusage[mem=${MEMORY}]" # span[hosts=1]"

export OMP_NUM_THREADS=${CORES}
export OPENBLAS_NUM_THREADS=${CORES}
export MKL_NUM_THREADS=${CORES}

module load cellgen/singularity

singularity exec \
  --bind ${READFORMER_DIR}:/scripts/readformer \
  --bind ${VALIDATION_OUTPUT_DIR}:/nst_dir \
  --pwd /scripts/readformer \
  ${SIF} \
  python3 /scripts/readformer/run_empirical_bayes.py \
    --fold ${FOLD} \
    --validation_output_dir /nst_dir \
    --desired_dfr 0.01
#    --random_seed 42  # use if you are subsampling the data to fit the gmm
#    --sample_size 200000  # use if you are subsampling the data to fit the gmm


EOF
  )
  if [[ -z "${job_id}" ]]; then
    echo "Job submission failed for ${MODEL_NAME} using fold ${FOLD}"
    exit 1
  fi

  echo "Submitted job ${job_id} for ${MODEL_NAME} using fold ${FOLD}"

done

