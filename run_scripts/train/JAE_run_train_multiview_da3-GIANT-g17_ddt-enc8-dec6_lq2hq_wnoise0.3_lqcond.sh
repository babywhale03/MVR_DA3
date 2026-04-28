
SERVER=20
NUM_GPUS=1
CUDA=3

export SERVER=${SERVER}
export CUDA=${CUDA}
export PATH="$CONDA_PREFIX/bin:$PATH"
export PYTHONBREAKPOINT=0

# stage2 training (stage1 - da3)
CUDA_VISIBLE_DEVICES=${CUDA} python -m torch.distributed.run --standalone --nproc_per_node=${NUM_GPUS} RAE/src/train_lq2hq.py \
  --config run_configs/train/JAE_run_train_multiview_da3-GIANT-g17_ddt-enc8-dec6_lq2hq_wnoise0.3_lqcond.yaml \