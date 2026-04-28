
SERVER=20
NUM_GPUS=1
CUDA=0

export SERVER=${SERVER}
export CUDA=${CUDA}
export PATH="$CONDA_PREFIX/bin:$PATH"
# stage2 training (stage1 - da3)
CUDA_VISIBLE_DEVICES=${CUDA} python -m torch.distributed.run --standalone --nproc_per_node=${NUM_GPUS} RAE/src/train_vae_lq2hq.py \
  --config run_configs/train/JAE_run_train_multiview_flux_ddt_lq2hq.yaml \