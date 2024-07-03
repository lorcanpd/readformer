#!/bin/bash
#BSUB -J memory_requirements_job
#BSUB -q gpu-normal
#BSUB -o logs/memory_requirements_%J.out
#BSUB -e logs/memory_requirements_%J.err
#BSUB -M 8192
#BSUB -n 4
#BSUB -gpu "mode=shared:num=1:gmem=40960"
#BSUB -R "select[mem>8192] rusage[mem=8192] span[hosts=1]"
#BSUB -W 00:20


# Removed from the above:
# -gpu "mode=shared:num=1:gmem=81920::gmodel=NVIDIAA100_SXM4_80GB"
READFORMER_DIR="/lustre/scratch126/casm/team274sb/lp23/readformer/readformer"
SYMLINK_DIR="/lustre/scratch126/casm/team274sb/lp23/readformer/data/pretrain_symlinks"
META_DATA_PATH="/lustre/scratch126/casm/team274sb/lp23/readformer/data/pretrain_metadata.csv"

echo "Loading Singularity"
module load cellgen/singularity

echo "Running memory_requirements.py for 2 layer Hyena model with 128 dim embeddings, kernel size 13, and 2 groups."
singularity exec --nv \
  --bind ${READFORMER_DIR}:/scripts/readformer \
  --bind ${SYMLINK_DIR}:/data/pretrain_symlinks \
  --bind ${META_DATA_PATH}:/data/pretrain_metadata.csv \
  --pwd /scripts/readformer \
  /nfs/users/nfs_l/lp23/sifs/readformer.sif \
  python3 /scripts/readformer/memory_requirements.py \
    --batch_size 64 \
    --emb_dim 128 \
    --max_sequence_length 8192 \
    --num_layers 2 \
    --hyena \
    --kernel_size 13 \
    --heads 2 \
    --data_dir /data/pretrain_symlinks \
    --metadata_path /data/pretrain_metadata.csv \
    --num_workers 4 \
    --prefetch_factor 2 \
    --min_quality 20 \
    --shuffle

if [ $? -ne 0 ]; then
  echo "Error running memory_requirements.py for 2 layer Hyena model with 128 dim embeddings, kernel size 13, and 2 groups."
fi

echo "Running memory_requirements.py for 4 layer Hyena model with kernel size 13 and 2 groups and 128 dim embeddings."
singularity exec --nv \
  --bind ${READFORMER_DIR}:/scripts/readformer \
  --bind ${SYMLINK_DIR}:/data/pretrain_symlinks \
  --bind ${META_DATA_PATH}:/data/pretrain_metadata.csv \
  --pwd /scripts/readformer \
  /nfs/users/nfs_l/lp23/sifs/readformer.sif \
  python3 /scripts/readformer/memory_requirements.py \
    --batch_size 64 \
    --emb_dim 128 \
    --max_sequence_length 8192 \
    --num_layers 4 \
    --hyena \
    --kernel_size 13 \
    --heads 2 \
    --data_dir /data/pretrain_symlinks \
    --metadata_path /data/pretrain_metadata.csv \
    --num_workers 4 \
    --prefetch_factor 2 \
    --min_quality 20 \
    --shuffle

if [ $? -ne 0 ]; then
  echo "Error running memory_requirements.py for 4 layer Hyena model with kernel size 13 and 2 groups and 128 dim embeddings."
fi

echo "Running memory_requirements.py for 4 layer Hyena model with kernel size 13 and 2 groups and 128 dim embeddings."
singularity exec --nv \
  --bind ${READFORMER_DIR}:/scripts/readformer \
  --bind ${SYMLINK_DIR}:/data/pretrain_symlinks \
  --bind ${META_DATA_PATH}:/data/pretrain_metadata.csv \
  --pwd /scripts/readformer \
  /nfs/users/nfs_l/lp23/sifs/readformer.sif \
  python3 /scripts/readformer/memory_requirements.py \
    --batch_size 64 \
    --emb_dim 256 \
    --max_sequence_length 8192 \
    --num_layers 4 \
    --hyena \
    --kernel_size 13 \
    --heads 2 \
    --data_dir /data/pretrain_symlinks \
    --metadata_path /data/pretrain_metadata.csv \
    --num_workers 4 \
    --prefetch_factor 2 \
    --min_quality 20 \
    --shuffle

if [ $? -ne 0 ]; then
  echo "Error running memory_requirements.py for 4 layer Hyena model with kernel size 13 and 2 groups and 128 dim embeddings."
fi

echo "Running memory_requirements.py for 2 layer Hyena model with kernel size 13 and 2 groups and 256 dim embeddings."
singularity exec --nv \
  --bind ${READFORMER_DIR}:/scripts/readformer \
  --bind ${SYMLINK_DIR}:/data/pretrain_symlinks \
  --bind ${META_DATA_PATH}:/data/pretrain_metadata.csv \
  --pwd /scripts/readformer \
  /nfs/users/nfs_l/lp23/sifs/readformer.sif \
  python3 /scripts/readformer/memory_requirements.py \
    --batch_size 32 \
    --emb_dim 256 \
    --max_sequence_length 8192 \
    --num_layers 2 \
    --hyena \
    --kernel_size 13 \
    --heads 2 \
    --data_dir /data/pretrain_symlinks \
    --metadata_path /data/pretrain_metadata.csv \
    --num_workers 4 \
    --prefetch_factor 2 \
    --min_quality 20 \
    --shuffle

if [ $? -ne 0 ]; then
  echo "Error running memory_requirements.py for 2 layer Hyena model with kernel size 13 and 2 groups and 256 dim embeddings."
fi

echo "Running memory_requirements.py for 2 layer transformer model with 8 heads and 128 dim embeddings."
singularity exec --nv \
  --bind ${READFORMER_DIR}:/scripts/readformer \
  --bind ${SYMLINK_DIR}:/data/pretrain_symlinks \
  --bind ${META_DATA_PATH}:/data/pretrain_metadata.csv \
  --pwd /scripts/readformer \
  /nfs/users/nfs_l/lp23/sifs/readformer.sif \
  python3 /scripts/readformer/memory_requirements.py \
    --batch_size 64 \
    --emb_dim 128 \
    --max_sequence_length 8192 \
    --num_layers 2 \
    --heads 8 \
    --data_dir /data/pretrain_symlinks \
    --metadata_path /data/pretrain_metadata.csv \
    --num_workers 4 \
    --prefetch_factor 2 \
    --min_quality 20 \
    --shuffle

if [ $? -ne 0 ]; then
  echo "Error running memory_requirements.py for 2 layer transformer model with 8 heads and 128 dim embeddings."
fi

echo "Running memory_requirements.py for 2 layer transformer model with 16 heads and 256 dim embeddings."
singularity exec --nv \
  --bind ${READFORMER_DIR}:/scripts/readformer \
  --bind ${SYMLINK_DIR}:/data/pretrain_symlinks \
  --bind ${META_DATA_PATH}:/data/pretrain_metadata.csv \
  --pwd /scripts/readformer \
  /nfs/users/nfs_l/lp23/sifs/readformer.sif \
  python3 /scripts/readformer/memory_requirements.py \
    --batch_size 32 \
    --emb_dim 256 \
    --max_sequence_length 8192 \
    --num_layers 2 \
    --heads 16 \
    --data_dir /data/pretrain_symlinks \
    --metadata_path /data/pretrain_metadata.csv \
    --num_workers 4 \
    --prefetch_factor 2 \
    --min_quality 20 \
    --shuffle

if [ $? -ne 0 ]; then
  echo "Error running memory_requirements.py for 2 layer transformer model with 16 heads and 256 dim embeddings."
fi

echo "Finished"


# python3 memory_requirements.py --batch_size 64 --emb_dim 128 --max_sequence_length 8192 --num_layers 4 --data_dir GIAB_BAM/illumina_2x250bps --metadata_path GIAB_BAM/pretraining_metadata.csv --num_workers 4 --prefetch_factor 2 --min_quality 20 --shuffle --kernel_size 13
