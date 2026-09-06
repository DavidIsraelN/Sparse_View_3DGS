#!/bin/bash

# Exit immediately if any command fails (except those we explicitly allow)
set -e

# ==============================================================================
# CONFIGURATION - get the constants from config file
# ==============================================================================
SCENE=$(python -c "from project_config import SCENE_NAME; print(SCENE_NAME)")
RESOLUTION=$(python -c "from project_config import RESOLUTION_SCALE; print(RESOLUTION_SCALE)")
ITERATIONS=$(python -c "from project_config import REGULAR_ITERATIONS; print(REGULAR_ITERATIONS)")
EX_A_ITERATIONS=$(python -c "from project_config import EXTENSION_A_ITERATIONS; print(EXTENSION_A_ITERATIONS)")
VOXEL=$(python -c "from project_config import VOXEL_SIZE; print(VOXEL_SIZE)")

# Pull the exact directory paths dynamically from our Python configuration!
SCENE_DIR=$(python -c "from project_config import SCENE_DIR; print(SCENE_DIR)")
ORIGINAL_IMAGES_DIR=$(python -c "from project_config import ORIGINAL_IMAGES_DIR; print(ORIGINAL_IMAGES_DIR)")
COLMAP_DIR=$(python -c "from project_config import COLMAP_DIR; print(COLMAP_DIR)")
COLMAP_IMAGES_DIR=$(python -c "from project_config import COLMAP_IMAGES_DIR; print(COLMAP_IMAGES_DIR)")

VANILLA_OUT=$(python -c "from project_config import VANILLA_OUT_DIR; print(VANILLA_OUT_DIR)")
SMDGS_OUT=$(python -c "from project_config import SMDGS_OUT_DIR; print(SMDGS_OUT_DIR)")

RUN_16_OUT=$(python -c "from project_config import RUN_16_OUT_DIR; print(RUN_16_OUT_DIR)")
RUN_17_OUT=$(python -c "from project_config import RUN_17_OUT_DIR; print(RUN_17_OUT_DIR)")
RUN_18_OUT=$(python -c "from project_config import RUN_18_OUT_DIR; print(RUN_18_OUT_DIR)")
RUN_20_OUT=$(python -c "from project_config import RUN_20_OUT_DIR; print(RUN_20_OUT_DIR)")

echo "========================================================="
echo "Starting Sparse-View Pipeline for scene: $SCENE"
echo "========================================================="

# ---------------------------------------------------------
# Step 1: Filter COLMAP data and prepare workspace
# ---------------------------------------------------------
echo -e "\n---> [Step 1/6] Filtering COLMAP data..."
python preprocessing/filter_colmap.py

# ---------------------------------------------------------
# Step 2: Generate and format Depth Priors
# ---------------------------------------------------------
echo -e "\n---> [Step 2/6] Generating Depth Priors..."
python preprocessing/prepare_depth_priors.py

# ---------------------------------------------------------
# Step 3: Generate pair.txt using the original script
# Note: This script deletes the images folder and crashes at the end.
# We append '|| true' to forcefully ignore the OpenCV crash.
# ---------------------------------------------------------
echo -e "\n---> [Step 3/6] Generating pair.txt (ignoring expected crash)..."
python colmap2mvsnet_acm.py --data_path $COLMAP_DIR --save_folder $COLMAP_DIR || true

# ---------------------------------------------------------
# Step 4: Restore the deleted images
# ---------------------------------------------------------
echo -e "\n---> [Step 4/6] Restoring deleted images to the colmap workspace..."
# Create the directory just in case it was entirely removed
mkdir -p $COLMAP_IMAGES_DIR
# Copy all images from the original source to the colmap workspace
cp -r $ORIGINAL_IMAGES_DIR/* $COLMAP_IMAGES_DIR
echo "Images restored successfully."

# ---------------------------------------------------------
# Step 5: Train the Baseline Models
# ---------------------------------------------------------
echo -e "\n---> [Step 5/6] "
echo -e "Starting Vanilla 3DGS Baseline Training..."

cd Vanilla_3DGS
python train.py \
    -s $COLMAP_DIR \
    --resolution $RESOLUTION \
    --iterations $ITERATIONS \
    --model_path $VANILLA_OUT
cd ..
# maybe render some images...

echo -e "Starting SMDGS Baseline Training..."

python train.py \
    -s $SCENE_DIR \
    --resolution $RESOLUTION \
    --iterations $ITERATIONS \
    --model_path $SMDGS_OUT

# render and score the output
python render.py \
    -m $SMDGS_OUT \
    --voxel_size $VOXEL

python metrics.py \
    -m $SMDGS_OUT

# ---------------------------------------------------------
# Step 6: Ablation Study - Running Proposed Configurations
# ---------------------------------------------------------
echo -e "\n---> [Step 6/6] Starting Ablation Study configurations..."

echo -e "\n---> [Run 16] Hybrid LPC + Empty Space + Dropout"
python train.py \
    -s $SCENE_DIR \
    --resolution $RESOLUTION \
    --iterations $ITERATIONS \
    --model_path $RUN_16_OUT \
    --use_lpc

python render.py -m $RUN_16_OUT --voxel_size $VOXEL
python metrics.py -m $RUN_16_OUT

echo -e "\n---> [Run 17] Hybrid LPC + Empty Space + GNS (3D Pruning, no Dropout)"
python train.py \
    -s $SCENE_DIR \
    --resolution $RESOLUTION \
    --iterations $ITERATIONS \
    --model_path $RUN_17_OUT \
    --use_lpc \
    --use_gns

python render.py -m $RUN_17_OUT --voxel_size $VOXEL
python metrics.py -m $RUN_17_OUT

echo -e "\n---> [Run 18] Hybrid LPC + Empty Space + GNS + Extension A (TV Loss)"
python train.py \
    -s $SCENE_DIR \
    --resolution $RESOLUTION \
    --iterations $EX_A_ITERATIONS \
    --geom_prior_until_iter $EX_A_ITERATIONS \
    --model_path $RUN_18_OUT \
    --use_lpc \
    --use_gns \
    --use_extension_a \
    --lambda_tv 0.1

python render.py -m $RUN_18_OUT --voxel_size $VOXEL
python metrics.py -m $RUN_18_OUT

echo -e "\n---> [Run 20] Hybrid LPC + Empty Space + Dropout + Extension A (No GNS)"
python train.py \
    -s $SCENE_DIR \
    --resolution $RESOLUTION \
    --iterations $EX_A_ITERATIONS \
    --geom_prior_until_iter $EX_A_ITERATIONS \
    --model_path $RUN_20_OUT \
    --use_lpc \
    --use_extension_a \
    --lambda_tv 0.1

python render.py -m $RUN_20_OUT --voxel_size $VOXEL
python metrics.py -m $RUN_20_OUT

echo -e "\n========================================================="
echo "Pipeline and Ablation Study completed successfully!"
echo "========================================================="
