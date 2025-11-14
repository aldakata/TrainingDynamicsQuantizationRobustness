#!/bin/bash
export FAST=/fast/atatjer
export SOFT_FILELOCK=1
export HF_HOME=/fast/atatjer/hf_fast          # if download
export HOME=/fast/atatjer/tmp
export UV_CACHE_DIR=/home/atatjer/.cache
export UV_PYTHON_INSTALL_DIR=/home/atatjer/.local/share/uv/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_DATASETS_TRUST_REMOTE_CODE=True
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TORCHINDUCTOR_DISABLE=1
# export FLASHINFER_DISABLE_JIT=1
# export VLLM_DISABLE_FLASHINFER=1
# export VLLM_USE_FLASHINFER_SAMPLER=0
module load cuda/12.9
module load gcc/9
echo $(printenv)
source /home/atatjer/src/scalinglawsquantization/.venv/bin/activate