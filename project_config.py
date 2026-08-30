"""
Global Configuration File for Sparse-View 3DGS Project
All paths and constants should be managed from this file to ensure consistency
across data preprocessing, training, and evaluation scripts.
"""

import os

# ==============================================================================
# 0. PROJECT ROOT ANCHOR
# Dynamically find the absolute path of the directory containing this config file.
# This ensures scripts work perfectly regardless of the terminal's current directory.
# ==============================================================================
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

# ==============================================================================
# 1. GLOBAL SCENE SETTINGS
# Change this single variable to switch the entire pipeline to a different scene
# ==============================================================================
SCENE_NAME = "bicycle"

# ==============================================================================
# 2. DEPTH MODEL CONFIGURATION
# ==============================================================================
MODEL_SIZE = 'vitb'  # Options: 'vits', 'vitb', 'vitl'
# Anchor the weights path to the project root
WEIGHTS_PATH = os.path.join(PROJECT_ROOT, "preprocessing", "depth_estimation", "weights", f"depth_anything_v2_{MODEL_SIZE}.pth")

# ==============================================================================
# 3. DIRECTORY PATHS (Dynamically generated based on SCENE_NAME)
# ==============================================================================
# Anchor the data root to the project root
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
SCENE_DIR = os.path.join(DATA_ROOT, SCENE_NAME)

# --- A. Original Source Paths ---
# Where you manually placed the extracted images and the original COLMAP binaries
ORIGINAL_IMAGES_DIR = os.path.join(SCENE_DIR, "images")
ORIGINAL_SPARSE_DIR = os.path.join(SCENE_DIR, "sparse", "0")

# --- B. SMDGS Workspace Paths ---
# The expected folder structure for the SMDGS codebase (-s flag points to COLMAP_DIR)
COLMAP_DIR = os.path.join(SCENE_DIR, "colmap")
COLMAP_IMAGES_DIR = os.path.join(COLMAP_DIR, "images")
COLMAP_SPARSE_DIR = os.path.join(COLMAP_DIR, "sparse", "0")

# --- C. Depth Priors Outputs ---
DEPTH_PRIORS_OUT = os.path.join(SCENE_DIR, "depth_priors")
TENSORS_DIR = os.path.join(DEPTH_PRIORS_OUT, "tensors")
VIS_DIR = os.path.join(DEPTH_PRIORS_OUT, "visualizations")

# --- D. SMDGS Required Target Directories ---
# SMDGS expects the .npy depth files exactly here
ALIGNED_DEPTH_DIR = os.path.join(SCENE_DIR, "output", "local_aligned")

# --- E. Training Outputs ---
OUTPUT_BASE_DIR = os.path.join(PROJECT_ROOT, "output")
VANILLA_OUT_DIR = os.path.join(OUTPUT_BASE_DIR, SCENE_NAME, "baseline_vanilla")
SMDGS_OUT_DIR = os.path.join(OUTPUT_BASE_DIR, SCENE_NAME, "baseline_smdgs")
OUR_METHOD_OUT_DIR = os.path.join(OUTPUT_BASE_DIR, SCENE_NAME, "selective_LPC_Hybrid_loss")

# =========================================================
# 4. Training Hyperparameters
# =========================================================
# Resolution scaling (1 = full, 2 = half, 4 = quarter)
RESOLUTION_SCALE = 2

# Total training iterations (SMDGS uses 15k for geometric priors)
TOTAL_ITERATIONS = 15000

# Spherical Harmonics degree (Default is 3. Use 1 or 0 to save massive VRAM)
SH_DEGREE = 3

# =========================================================
def ensure_directories():
    """
    Utility function to verify that required output directories exist.
    """
    os.makedirs(VANILLA_OUT_DIR, exist_ok=True)
    os.makedirs(SMDGS_OUT_DIR, exist_ok=True)
    os.makedirs(OUR_METHOD_OUT_DIR, exist_ok=True)
    
    # Ensure COLMAP workspace directories exist
    os.makedirs(COLMAP_IMAGES_DIR, exist_ok=True)
    os.makedirs(COLMAP_SPARSE_DIR, exist_ok=True)
    
    # Ensure depth directories exist
    os.makedirs(TENSORS_DIR, exist_ok=True)
    os.makedirs(VIS_DIR, exist_ok=True)
    os.makedirs(ALIGNED_DEPTH_DIR, exist_ok=True)

# Run this check automatically when the config is imported
ensure_directories()
