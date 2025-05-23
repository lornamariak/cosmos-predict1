# Setting Up `cosmos-predict1` Post-Training Environment

Start by building the `Dockerfile` here -`Dockerfile2`.

**Note:** It will take a while to build the APEX library; ensure you don't time out.

## Once Inside the Environment

Follow these instructions from the`cosmos-predict1`post-training setup with a few modifications.

### 1. Log in to Hugging Face CLI
This is necessary for downloading models and datasets.
```bash
huggingface-cli login
```

### 2. Download Diffusion Model Checkpoints
This script downloads pre-trained model weights for 7B and 14B parameter models of type `Text2World`.
`CUDA_HOME` and `PYTHONPATH` are set to ensure the script can find CUDA and project-specific modules.
```bash
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python scripts/download_diffusion_checkpoints.py --model_sizes 7B 14B --model_types Text2World
```

### 3. Download Cosmos NeMo Assets Dataset
This downloads only `.mp4` files from the `nvidia/Cosmos-NeMo-Assets` dataset on Hugging Face into the specified local directory.
```bash
huggingface-cli download nvidia/Cosmos-NeMo-Assets --repo-type dataset --local-dir datasets/cosmos_nemo_assets/ --include "*.mp4*"
```

### 4. Rename Downloaded Data
Rename the downloaded example data directory to `videos`.
```bash
mv datasets/cosmos_nemo_assets/nemo_diffusion_example_data datasets/cosmos_nemo_assets/videos
```

### 5. Generate T5 Embeddings
This script processes videos in the `dataset_path` and generates embeddings based on the provided prompt.
```bash
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python scripts/get_t5_embeddings_from_cosmos_nemo_assets.py --dataset_path datasets/cosmos_nemo_assets --prompt "A video of sks teal robot."
```

### 6. Check NVRTC Library
Check for the NVIDIA CUDA Runtime Compilation (NVRTC) library.
- `ldconfig -p` lists the shared library cache.
- `conda list | grep cuda` lists conda packages related to CUDA.
- `find / -name "libnvrtc*"` finds all files named `libnvrtc*` in the entire filesystem (errors are redirected to `/dev/null`).

```bash
ldconfig -p | grep libnvrtc
conda list | grep cuda
find / -name "libnvrtc*" 2>/dev/null
```

### 7. Configure PyTorch CUDA Memory Allocation
- `expandable_segments:True` helps to avoid fragmentation.
- `max_split_size_mb:128` limits the size of memory blocks PyTorch splits, potentially reducing fragmentation.
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
```

### 8. Set `LD_LIBRARY_PATH` and Create Symbolic Links
Set `LD_LIBRARY_PATH` to include paths to CUDA libraries within the conda environment. This helps the system find necessary shared libraries.
```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/conda/envs/cosmos-predict1/lib:/opt/conda/envs/cosmos-predict1/targets/x86_64-linux/lib
```
Create symbolic links for NVRTC libraries from the conda environment to system library paths. This can resolve issues where applications expect these libraries in standard system locations.
```bash
ln -sf /opt/conda/envs/cosmos-predict1/lib/libnvrtc.so /usr/lib/libnvrtc.so
ln -sf /opt/conda/envs/cosmos-predict1/lib/libnvrtc.so.12 /usr/lib/libnvrtc.so.12
ln -sf /opt/conda/envs/cosmos-predict1/lib/libnvrtc-builtins.so /usr/lib/libnvrtc-builtins.so
```
Update the shared library cache after creating new symbolic links.
```bash
ldconfig
```
Verify that the NVRTC library is now correctly found by the linker.
```bash
ldconfig -p | grep libnvrtc
```

### 9. Run Training
Run training, overriding the maximum iterations.
`trainer.max_iter=100` sets the training to stop after 100 iterations.
```bash
torchrun --nproc_per_node=4 -m cosmos_predict1.diffusion.training.train \
    --config=cosmos_predict1/diffusion/training/config/config.py \
    -- experiment=text2world_7b_example_cosmos_nemo_assets trainer.max_iter=100
```