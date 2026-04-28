export PATH="$CONDA_PREFIX/bin:$PATH"

time CUDA_VISIBLE_DEVICES=7 python -m depth_anything_3.bench.evaluator_lq2hq_scene_json --config run_configs/DA3/val/JAE__w-mvrm-FRONT__da3-giant__ddt-enc8-dec6__lq2hq_lqcond__scene_json.yaml
