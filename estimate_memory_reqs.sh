#!/bin/bash

#BSUB -J memory_requirements_job
#BSUB -q gpu-normal
#BSUB -o logs/memory_requirements_%J.out
#BSUB -e logs/memory_requirements_%J.err
#BSUB -M 51200
#BSUB -n 4
#BSUB -gpu "num=1:gmem=10240"
#BSUB -R "select[mem>10240] rusage[mem=10240, ngpus_physical=1, gmem=10240] span[hosts=1]"
#BSUB -W 00:30

echo "Loading Singularity"
module load cellgen/singularity

echo "Running memory_requirements.py"
singularity exec --nv /nfs/users/nfs_l/lp23/sifs/readformer.sif \
  python3 memory_requirements.py

echo "Finished"