#!/bin/bash

# Exit immediately if any command fails
set -e

echo "========================================================="
echo "Setting up environment for Sparse-View 3DGS"
echo "========================================================="

# 1. Update system C/C++ compilers (Requires sudo privileges)
echo "--> Updating system build tools (C/C++ compilers)..."
sudo apt update
sudo apt install build-essential -y

# 2. Initialize Conda for this script and create the environment
echo "--> Initializing Conda and creating environment 'sparse_3dgs'..."
# This magical line ensures 'conda activate' works inside a bash script
eval "$(conda shell.bash hook)"

# Create the environment with Python 3.10 (using -y to auto-approve)
conda create -n sparse_3dgs python=3.10 -y

# Activate the newly created environment
echo "--> Activating environment..."
conda activate sparse_3dgs

# 3. Install CUDA Toolkit directly into the Conda environment
echo "--> Installing CUDA Toolkit 12.1 via Conda..."
conda install -c "nvidia/label/cuda-12.1.0" cuda-toolkit -y

# 4. Install PyTorch matching the exact CUDA version
echo "--> Installing PyTorch for CUDA 12.1..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 5. Install standard Python requirements
echo "--> Installing standard Python packages from requirements.txt..."
pip install -r requirements.txt

# 6. Install PyTorch3D from source (crucial to do this AFTER PyTorch is installed)
echo "--> Compiling PyTorch3D from source (this may take a while)..."
pip install "git+https://github.com/facebookresearch/pytorch3d.git" --no-build-isolation

# 7. Compile SMDGS submodules
echo "--> Compiling SMDGS specific submodules..."
pip install ./submodules/diff-plane-rasterization --no-build-isolation
pip install ./submodules/simple-knn --no-build-isolation
pip install ./submodules/visible-detection --no-build-isolation

# 8. Compile Vanilla 3DGS submodules
echo "--> Compiling Vanilla 3DGS specific submodules..."
pip install ./Vanilla_3DGS/submodules/diff-gaussian-rasterization --no-build-isolation
pip install ./Vanilla_3DGS/submodules/fused-ssim --no-build-isolation
pip install ./Vanilla_3DGS/submodules/simple-knn --no-build-isolation

# 9. Download DepthAnythingV2 Weights
echo "--> Downloading DepthAnythingV2 (ViT-B) weights..."
mkdir -p preprocessing/depth_estimation/weights
wget -nc https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth -O preprocessing/depth_estimation/weights/depth_anything_v2_vitb.pth

echo "========================================================="
echo "Environment setup complete! To start working, simply run:"
echo "conda activate sparse_3dgs"
echo "========================================================="
