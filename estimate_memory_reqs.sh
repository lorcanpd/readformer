#!/bin/bash
#BSUB -J memory_requirements_job
#BSUB -q gpu-normal
#BSUB -o logs/memory_requirements_%J.out
#BSUB -e logs/memory_requirements_%J.err
#BSUB -M 51200
#BSUB -n 4
#BSUB -gpu "num=1:gmem=40960"
#BSUB -R "select[mem>51200] rusage[mem=51200, ngpus_physical=1, gmem=40960] span[hosts=1]"
#BSUB -W 00:15


echo "Loading Singularity"
module load cellgen/singularity

echo "Running memory_requirements.py"
singularity exec --nv /nfs/users/nfs_l/lp23/sifs/readformer.sif \
  python3 ${READFORMER_DIR}/memory_requirements.py

echo "Finished"
