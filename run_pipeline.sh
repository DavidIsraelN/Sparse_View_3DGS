#!/bin/bash

# Exit immediately if any command fails (except those we explicitly allow)
set -e

# ==============================================================================
# CONFIGURATION - get the constants from config file
# ==============================================================================
SCENE=$(python -c "from project_config import SCENE_NAME; print(SCENE_NAME)")
RESOLUTION=$(python -c "from project_config import RESOLUTION_SCALE; print(RESOLUTION_SCALE)")
ITERATIONS=$(python -c "from project_config import TOTAL_ITERATIONS; print(TOTAL_ITERATIONS)")

# Pull the exact directory paths dynamically from our Python configuration!
SCENE_DIR=$(python -c "from project_config import SCENE_DIR; print(SCENE_DIR)")
ORIGINAL_IMAGES_DIR=$(python -c "from project_config import ORIGINAL_IMAGES_DIR; print(ORIGINAL_IMAGES_DIR)")
COLMAP_DIR=$(python -c "from project_config import COLMAP_DIR; print(COLMAP_DIR)")
COLMAP_IMAGES_DIR=$(python -c "from project_config import COLMAP_IMAGES_DIR; print(COLMAP_IMAGES_DIR)")
VANILLA_OUT=$(python -c "from project_config import VANILLA_OUT_DIR; print(VANILLA_OUT_DIR)")
SMDGS_OUT=$(python -c "from project_config import SMDGS_OUT_DIR; print(SMDGS_OUT_DIR)")
OUR_METHOD_OUT=$(python -c "from project_config import OUR_METHOD_OUT_DIR; print(OUR_METHOD_OUT_DIR)")

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

# ---------------------------------------------------------
# Step 6: Train Our Proposed Model (Selective GAL)
# ---------------------------------------------------------
echo -e "\n---> [Step 6/6] "
echo -e "Starting Proposed Method (Selective Gradient-Alignment Loss) Training..."

# Execute the training script with the --use_gal flag to activate our custom loss
python train.py \
    -s $SCENE_DIR \
    --resolution $RESOLUTION \
    --iterations $ITERATIONS \
    --model_path $OUR_METHOD_OUT \
    --use_gal

echo -e "\n========================================================="
echo "Pipeline completed successfully!"
echo "========================================================="
