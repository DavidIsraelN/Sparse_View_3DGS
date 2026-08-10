import os
import glob
import cv2
import torch
import numpy as np
from preprocessing.depth_estimation.Depth_Anything_V2.depth_anything_v2.dpt import DepthAnythingV2

def generate_depth_priors(image_dir, output_dir, weights_path, model_size='vits'):
    # Create two subdirectories: one for math tensors (.pt) and one for visual images (.png)
    tensor_dir = os.path.join(output_dir, 'tensors')
    vis_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(tensor_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    
    # Check if GPU is available to utilize the RTX 4050
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
    depth_anything.load_state_dict(torch.load(weights_path, map_location='cpu'))
    depth_anything = depth_anything.to(device).eval()
    
    image_paths = glob.glob(os.path.join(image_dir, '*.JPG')) + glob.glob(os.path.join(image_dir, '*.png'))

    with torch.no_grad():
        for img_path in image_paths:
            print(f"Processing: {img_path}")
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
            tensor_path = os.path.join(tensor_dir, f"depth_{name_without_ext}.pt")
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
            
            vis_path = os.path.join(vis_dir, f"vis_{name_without_ext}.png")
            cv2.imwrite(vis_path, colored_depth)
            
            print(f"Saved tensor to {tensor_path} and visualization to {vis_path}")


if __name__ == '__main__':
    # Define the input and output directories
    IMG_DIR = "data/bicycle/images"
    OUT_DIR = "data/bicycle/depth_priors"
    
    # ---------------------------------------------------------
    # Configuration to switch between model sizes
    # ---------------------------------------------------------
    # Change MODEL_SIZE to 'vitb' or 'vitl' if you downloaded larger weights
    # MODEL_SIZE = 'vits' 
    # WEIGHTS = "preprocessing/depth_estimation/weights/depth_anything_v2_vits.pth"
    MODEL_SIZE = 'vitb' 
    WEIGHTS = "preprocessing/depth_estimation/weights/depth_anything_v2_vitb.pth"
    
    generate_depth_priors(IMG_DIR, OUT_DIR, WEIGHTS, model_size=MODEL_SIZE)
    