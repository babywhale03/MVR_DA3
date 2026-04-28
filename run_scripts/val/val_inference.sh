SERVER=20
NUM_GPUS=1
CUDA=0

export SERVER=${SERVER}
export CUDA=${CUDA}
export PATH="$CONDA_PREFIX/bin:$PATH"
export PYTHONBREAKPOINT=0

CUDA_VISIBLE_DEVICES=${CUDA} python -m torch.distributed.run --standalone --nproc_per_node=${NUM_GPUS} RAE/src/inference.py \
    --config /mnt/dataset1/jaeeun/nips26/MVR/run_configs/train/JAE_run_train_multiview_da3-GIANT-g17_ddt-enc8-dec6_lq2hq_wnoise0.3.yaml \
    --ckpt /mnt/dataset1/jaeeun/nips26/MVR/result_train/nips26/multi_view/260402_test_lq2hq_wnoise0.3_enc8-dec6/hypersim/TRAIN__fp32__hypersim__near_camera-near_random__da3-GIANT-extractfeat17-mvrm__bs1-maxview4-accum1__lr-2e-04__msg-lqkernel200__ema95__hyp1__ddt-enc8-1536-dec6-3072__lqdrop01_re__concat__lq2hq_wnoise0.3/checkpoints/ep-0050000.pt \
    --save_dir ./onestep_vis \
    --num_timesteps 20