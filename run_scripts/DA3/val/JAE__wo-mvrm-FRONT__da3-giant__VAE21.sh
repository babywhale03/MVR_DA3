export PATH="$CONDA_PREFIX/bin:$PATH"
export PYTHONPATH=/mnt/dataset1/jaeeun/nips26/MVR/Depth-Anything-3/src:$PYTHONPATH
# export PYTHONBREAKPOINT=0

time CUDA_VISIBLE_DEVICES=0 python -m depth_anything_3.bench.evaluator --config run_configs/DA3/val/JAE__wo-mvrm__da3-giant__VAE21.yaml
