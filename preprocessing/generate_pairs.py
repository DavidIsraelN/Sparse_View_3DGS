import os
import glob

def create_dummy_pair_txt(images_dir, output_path):
    # find all the images in the folder and sort them by alpha-beta
    extensions = ('*.JPG', '*.jpg', '*.png', '*.PNG')
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(images_dir, ext)))
    image_paths = sorted(image_paths)
    
    num_images = len(image_paths)
    if num_images == 0:
        print("No images found in the specified directory!")
        return

    print(f"Found {num_images} images. Generating pair.txt...")
    
    with open(output_path, 'w') as f:
        # first line is the number of images
        f.write(f"{num_images}\n")
        
        for i in range(num_images):
            # line with idx of the current image
            f.write(f"{i}\n")
            
            # Create neighbor list: number of neighbors, then pairs of (neighbor index, match score)
            # The match score (2000.0) is completely arbitrary and is only intended to pass the code's filter
            neighbors = []
            for j in range(num_images):
                if i != j:
                    neighbors.append(f"{j} 2000.0")
            
            num_neighbors = len(neighbors)
            neighbors_str = " ".join(neighbors)
            f.write(f"{num_neighbors} {neighbors_str}\n")
            
    print(f"Successfully created pair.txt at {output_path}")

if __name__ == '__main__':

    img_dir = "../data/bicycle/colmap/images"
    out_file = "../data/bicycle/colmap/pair.txt"

    create_dummy_pair_txt(img_dir, out_file)
