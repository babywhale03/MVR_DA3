export PATH="$CONDA_PREFIX/bin:$PATH"

time CUDA_VISIBLE_DEVICES=6 python -m depth_anything_3.bench.evaluator_lq2hq --config run_configs/DA3/val/JAE__w-mvrm-FRONT__da3-giant__ddt-enc8-dec6__lq2hq_wnoise0.7.yaml
