FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV VENV_PATH="/env"
ENV PATH="${VENV_PATH}/bin:${PATH}"

# Install Python and necessary tools
RUN apt update && \
    apt install --no-install-recommends --yes python3.11 python3.11-venv python3.11-dev python3-pip && \
    apt-get clean && \
    ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    ln -sf /usr/bin/pip3 /usr/bin/pip && \
    rm -rf /var/lib/apt/lists/*

# Create and activate a virtual environment, then install the specified packages
RUN python3 -m venv $VENV_PATH && \
    . ${VENV_PATH}/bin/activate && \
    pip install --upgrade pip wheel setuptools && \
    pip install torch==2.3.0+cu121 torchvision==0.18.0+cu121 torchaudio==2.3.0+cu121 -f https://download.pytorch.org/whl/torch_stable.html && \
    pip install pysam==0.22.0 matplotlib==3.8.4 wandb==0.17.2 seaborn==0.13.2 pandas==2.2.1 ipython

# Create directories for the user to mount data and code
RUN mkdir -p /nfs /nst_dir /data/pretrain_symlinks /data/finetune/BAM /data/finetune/VCF /scripts/readformer

# Provide an entry point that does nothing, allowing the user to define commands at runtime
ENTRYPOINT ["/bin/bash"]
