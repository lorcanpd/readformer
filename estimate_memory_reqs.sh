#!/bin/bash
#BSUB -J memory_requirements_job
#BSUB -q gpu-normal
#BSUB -o logs/memory_requirements_%J.out
#BSUB -e logs/memory_requirements_%J.err
#BSUB -M 40960
#BSUB -n 4
#BSUB -gpu "mode=shared:num=1:gmem=40960"
#BSUB -R "select[mem>40960] rusage[mem=40960] span[hosts=1]"
#BSUB -W 00:15


# Removed from the above:
# -gpu "mode=shared:num=1:gmem=81920::gmodel=NVIDIAA100_SXM4_80GB"
READFORMER_DIR="/lustre/scratch126/casm/team274sb/lp23/readformer/readformer"
SYMLINK_DIR="/lustre/scratch126/casm/team274sb/lp23/readformer/data/pretrain_symlinks"
META_DATA_PATH="/lustre/scratch126/casm/team274sb/lp23/readformer/data/pretrain_metadata.csv"

echo "Loading Singularity"
module load cellgen/singularity

echo "Running memory_requirements.py"
singularity exec --nv \
  --bind ${READFORMER_DIR}:/scripts/readformer \
  --bind ${SYMLINK_DIR}:/data/pretrain_symlinks \
  --bind ${META_DATA_PATH}:/data/pretrain_metadata.csv \
  --pwd /scripts/readformer \
  /nfs/users/nfs_l/lp23/sifs/readformer.sif \
  python3 /scripts/readformer/memory_requirements.py \
    --batch_size 256 \
    --emb_dim 512 \
    --max_sequence_length 8192 \
    --num_layers 6 \
    --heads 16 \
    --data_dir /data/pretrain_symlinks \
    --metadata_path /data/pretrain_metadata.csv \
    --num_workers 4 \
    --prefetch_factor 2 \
    --min_quality 20 \
    --shuffle


echo "Finished"
