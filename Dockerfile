#FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV VENV_PATH="/env"
ENV PATH="${VENV_PATH}/bin:${PATH}"

# Install Python and necessary tools
RUN apt update && \
    apt install --no-install-recommends --yes python3.11 python3.11-venv python3.11-dev python3-pip && \
    apt install --no-install-recommends --yes bcftools && \
    apt-get clean && \
    ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    ln -sf /usr/bin/pip3 /usr/bin/pip && \
    rm -rf /var/lib/apt/lists/*

# Create and activate a virtual environment, then install the specified packages
#    pip install torch==2.7.0+cu121 torchvision==0.18.0+cu121 torchaudio==2.3.0+cu121 -f https://download.pytorch.org/whl/torch_stable.html && \

RUN python3 -m venv $VENV_PATH && \
    . ${VENV_PATH}/bin/activate && \
    pip install --upgrade pip wheel setuptools && \
    pip install torch==2.7.1+cu118 torchvision==0.22.1+cu118 torchaudio==2.7.1+cu118 --index-url https://download.pytorch.org/whl/cu118 && \
    pip install torchrl==0.8.1 && \
    pip install pysam==0.22.0 matplotlib==3.8.4 wandb==0.17.2 seaborn==0.13.2 pandas==2.2.1 ipython \
        SigProfilerAssignment torchmetrics==1.6.0 tabulate scikit-learn

# Create directories for the user to mount data and code
RUN mkdir -p /nfs /nst_dir /data/pretrain/BAM /data/finetune/BAM /data/finetune/VCF /scripts/readformer \
    /models /home

# Provide an entry point that does nothing, allowing the user to define commands at runtime
ENTRYPOINT ["/bin/bash"]
