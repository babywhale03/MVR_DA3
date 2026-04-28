export PATH="$CONDA_PREFIX/bin:$PATH"
export PYTHONBREAKPOINT=0

time CUDA_VISIBLE_DEVICES=5 python -m depth_anything_3.bench.evaluator --config run_configs/DA3/val/JAE__w-mvrm-FRONT__da3-giant__ddt-enc4-dec8.yaml
