import os
import sys
import glob
from pathlib import Path

# Resolves the path dynamically to the project root (Sparse_View_3DGS)
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Import the centralized configuration variables
from project_config import COLMAP_IMAGES_DIR, COLMAP_DIR

def create_dummy_pair_txt(images_dir, output_path):
    """
    Generates a dummy pair.txt file required by the SMDGS pipeline.
    Assigns arbitrary high match scores to simulate nearest neighbors
    for a sparse-view setup, bypassing the MVSNet filtering logic.
    """
    # Find all the images in the folder and sort them alphabetically
    extensions = ('*.JPG', '*.jpg', '*.png', '*.PNG')
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(images_dir, ext)))
    image_paths = sorted(image_paths)
    
    num_images = len(image_paths)
    if num_images == 0:
        print(f"Error: No images found in the specified directory: {images_dir}")
        return

    print(f"Found {num_images} images. Generating pair.txt...")
    
    with open(output_path, 'w') as f:
        # First line is the total number of images
        f.write(f"{num_images}\n")
        
        for i in range(num_images):
            # Line with the index of the current image
            f.write(f"{i}\n")
            
            # Create neighbor list: number of neighbors, then pairs of (neighbor index, match score)
            # The match score (2000.0) is completely arbitrary and guarantees passing the code's threshold
            neighbors = []
            for j in range(num_images):
                if i != j:
                    neighbors.append(f"{j} 2000.0")
            
            num_neighbors = len(neighbors)
            neighbors_str = " ".join(neighbors)
            f.write(f"{num_neighbors} {neighbors_str}\n")
            
    print(f"Successfully created dummy pair.txt at {output_path}")

if __name__ == '__main__':
    # Define the output path directly inside the COLMAP working directory
    out_file = os.path.join(COLMAP_DIR, "pair.txt")
    
    # Run the generation process
    create_dummy_pair_txt(COLMAP_IMAGES_DIR, out_file)
    