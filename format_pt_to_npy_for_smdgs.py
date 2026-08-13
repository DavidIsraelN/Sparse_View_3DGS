import os
import torch
import numpy as np
import glob


def convert_pt_to_npy(pt_dir, out_dir):
    # Ensure the output directory exists
    os.makedirs(out_dir, exist_ok=True)
    
    # Retrieve all PyTorch tensor files from the input directory
    pt_files = glob.glob(os.path.join(pt_dir, "*.pt"))
    
    if not pt_files:
        print(f"No .pt files found in {pt_dir}")
        return

    for pt_file in pt_files:
        # Load the PyTorch tensor from disk
        tensor = torch.load(pt_file ,weights_only=True)
        
        # Convert the PyTorch tensor to a NumPy array
        numpy_arr = tensor.numpy()
        
        # Add a channel dimension if the array is purely 2D (H, W) -> (H, W, 1)
        # This prevents crashes during the transpose operation in SMDGS dataset_readers.py
        if len(numpy_arr.shape) == 2:
            numpy_arr = np.expand_dims(numpy_arr, axis=-1)
        
        # Format the filename to match the original image names exactly
        # e.g., "depth_DSC8681.pt" -> "_DSC8681.npy"
        base_name = os.path.basename(pt_file)
        new_name = base_name.replace("depth", "").replace(".pt", ".npy")
        out_path = os.path.join(out_dir, new_name)
        
        # Save the array in the standard NumPy binary format
        np.save(out_path, numpy_arr)
        print(f"Successfully converted and saved: {out_path}")


if __name__ == '__main__':
    # Define input and output directories matching the required folder structure
    INPUT_DIR = "data/bicycle/depth_priors/tensors"
    OUTPUT_DIR = "data/bicycle/output/local_aligned"
    
    convert_pt_to_npy(INPUT_DIR, OUTPUT_DIR)
