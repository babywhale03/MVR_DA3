## 📦 Installation 

### 1. Conda Environment

```bash
conda create -n mvrvggt python=3.10 -y
conda activate mvrvggt
```

### 2. Library Installation
```bash
# download torch
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121

# download other libraries
pip install -r requirements.txt

# download da3
cd Depth-Anything-3/
pip install -e . 
pip install --no-build-isolation --config-settings editable_mode=compat git+https://github.com/nerfstudio-project/gsplat.git@0b4dddf04cb687367602c01196913cde6a743d70
```

## 🔥 Training 

### 1. Training bash script
```bash
# CUDA=0,1,2,3,4,5,6,7
bash run_scripts/train/JIHYE_run_train_multiview_flux_ddt_lq2hq.sh
```

### 2. Training config yaml file : [MVR_DA3](run_configs/train/JIHYE_run_train_multiview_flux_ddt_lq2hq.yaml)