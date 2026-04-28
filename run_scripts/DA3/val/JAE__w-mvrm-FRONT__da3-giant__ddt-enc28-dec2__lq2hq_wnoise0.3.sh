export PATH="$CONDA_PREFIX/bin:$PATH"

time CUDA_VISIBLE_DEVICES=2 python -m depth_anything_3.bench.evaluator_lq2hq --config run_configs/DA3/val/JAE__w-mvrm-FRONT__da3-giant__ddt-enc28-dec2__lq2hq_wnoise0.3.yaml
