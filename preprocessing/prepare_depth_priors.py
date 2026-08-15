import os
import sys
import glob
import cv2
import torch
import numpy as np
from pathlib import Path
# Adds the main folder (Sparse_View_3DGS) to Python's path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from project_config import (
    ORIGINAL_IMAGES_DIR, TENSORS_DIR, VIS_DIR, 
    ALIGNED_DEPTH_DIR, MODEL_SIZE, WEIGHTS_PATH
)
from preprocessing.depth_estimation.Depth_Anything_V2.depth_anything_v2.dpt import DepthAnythingV2


def generate_depth_priors(image_dir, tensor_dir, vis_dir, weights_path, model_size='vits'):
    """
    Generates depth maps using DepthAnythingV2.
    Saves mathematical tensors (.pt) for PyTorch optimization and colored PNGs for visualization.
    """

    print(f"--- Phase 1: Generating Depth Priors ({model_size}) ---")
    os.makedirs(tensor_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    
    # Check if GPU is available to utilize the hardware acceleration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running on device: {device}")

    # Dictionary containing configurations for all DepthAnything V2 model sizes
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
    }
    
    print(f"Loading DepthAnythingV2 ({model_size}) model...")
    # Initialize the model architecture dynamically based on the chosen size
    depth_anything = DepthAnythingV2(**model_configs[model_size])
    
    # Load the corresponding pre-trained weights
    depth_anything.load_state_dict(torch.load(weights_path, map_location='cpu', weights_only=True))
    depth_anything = depth_anything.to(device).eval()
    
    image_paths = glob.glob(os.path.join(image_dir, '*.JPG')) + glob.glob(os.path.join(image_dir, '*.png'))

    with torch.no_grad():
        for img_path in image_paths:
            print(f"Processing image: {os.path.basename(img_path)}")
            # Read the image using OpenCV
            raw_image = cv2.imread(img_path)

            # Predict the depth map
            # The output is a 2D numpy array representing relative depth
            depth_map = depth_anything.infer_image(raw_image)

            base_name = os.path.basename(img_path)
            name_without_ext = os.path.splitext(base_name)[0]

            # ---------------------------------------------------------
            # 1. Save the exact mathematical tensor for 3DGS training
            # ---------------------------------------------------------
            depth_tensor = torch.from_numpy(depth_map).float()
            tensor_path = os.path.join(tensor_dir, f"depth{name_without_ext}.pt")
            torch.save(depth_tensor, tensor_path)

            # ---------------------------------------------------------
            # 2. Generate and save a visual representation (PNG)
            # ---------------------------------------------------------
            # Normalize the float array to a 0-1 range
            depth_min = depth_map.min()
            depth_max = depth_map.max()
            normalized_depth = (depth_map - depth_min) / (depth_max - depth_min)

            # Convert to an 8-bit unsigned integer array (0-255)
            depth_8bit = (normalized_depth * 255).astype(np.uint8)

            # Apply INFERNO colormap for better visual distinction of depth layers
            colored_depth = cv2.applyColorMap(depth_8bit, cv2.COLORMAP_INFERNO)

            vis_path = os.path.join(vis_dir, f"vis{name_without_ext}.png")
            cv2.imwrite(vis_path, colored_depth)

            print(f"Saved tensor and visualization for {os.path.basename(img_path)}")


def convert_pt_to_npy(pt_dir, out_dir):
    """
    Converts raw PyTorch tensors into NumPy arrays with an added channel dimension.
    This fulfills the strict format requirements of the SMDGS dataset loader.
    """
    print("\n--- Phase 2: Formatting Tensors for SMDGS Baseline ---")
    os.makedirs(out_dir, exist_ok=True)

    # Retrieve all PyTorch tensor files from the input directory
    pt_files = glob.glob(os.path.join(pt_dir, "*.pt"))

    if not pt_files:
        print(f"Error: No .pt files found in {pt_dir}")
        return

    for pt_file in pt_files:
        # Load the PyTorch tensor from disk securely
        tensor = torch.load(pt_file, weights_only=True)

        # Convert the PyTorch tensor to a NumPy array
        numpy_arr = tensor.numpy()

        # Add a channel dimension if the array is purely 2D (H, W) -> (H, W, 1)
        # This prevents crashes during the transpose operation in SMDGS dataset_readers.py
        if len(numpy_arr.shape) == 2:
            numpy_arr = np.expand_dims(numpy_arr, axis=-1)

        # Format the filename to match the original image names exactly
        # e.g., "depth_DSC8681.pt" -> "_DSC8681.npy" or "DSC8681.npy"
        base_name = os.path.basename(pt_file)
        new_name = base_name.replace("depth", "").replace(".pt", ".npy")
        out_path = os.path.join(out_dir, new_name)

        # Save the array in the standard NumPy binary format
        np.save(out_path, numpy_arr)
        print(f"Converted and saved target NPY for {new_name}")


if __name__ == '__main__':
    # Execute Phase 1: Model inference and tensor creation
    generate_depth_priors(ORIGINAL_IMAGES_DIR, 
                          TENSORS_DIR, 
                          VIS_DIR, 
                          WEIGHTS_PATH, 
                          model_size=MODEL_SIZE)
    
    # Execute Phase 2: Format conversion for SMDGS
    convert_pt_to_npy(TENSORS_DIR, ALIGNED_DEPTH_DIR)
    
    print("\nDepth priors preparation completed successfully!")
