import struct
import os
import glob


def filter_images_bin(input_bin_path, output_bin_path, valid_image_names):
    """
    Reads a COLMAP images.bin file, filters out cameras that are not in valid_image_names,
    and writes a new valid images.bin file.
    """
    images = []
    
    print(f"Reading original COLMAP binary: {input_bin_path}")
    with open(input_bin_path, "rb") as fid:
        # Read the number of registered images (Unsigned long long - 8 bytes)
        num_reg_images = struct.unpack("<Q", fid.read(8))[0]

        for i in range(num_reg_images):
            # Read image attributes
            image_id = struct.unpack("<I", fid.read(4))[0]
            qw, qx, qy, qz = struct.unpack("<dddd", fid.read(32))
            tx, ty, tz = struct.unpack("<ddd", fid.read(24))
            camera_id = struct.unpack("<I", fid.read(4))[0]
            
            # Read the image name (null-terminated string)
            name = b""
            while True:
                char = fid.read(1)
                if char == b"\x00":
                    break
                name += char
            name = name.decode("utf-8")
            
            # Read 2D points observations
            num_points2D = struct.unpack("<Q", fid.read(8))[0]
            # Each 2D point is: x (double), y (double), point3D_id (long long) = 24 bytes
            x_y_id_s = struct.unpack("<" + "ddq" * num_points2D, fid.read(24 * num_points2D))

            # Keep only the images that exist in our sparse folder
            if name in valid_image_names:
                images.append((image_id, qw, qx, qy, qz, tx, ty, tz, camera_id, name, num_points2D, x_y_id_s))
                print(f"Kept camera parameters for: {name}")

    print(f"Writing filtered COLMAP binary to: {output_bin_path}")
    with open(output_bin_path, "wb") as fid:
        # Write the new number of images
        fid.write(struct.pack("<Q", len(images)))

        for img in images:
            fid.write(struct.pack("<I", img[0]))
            fid.write(struct.pack("<dddd", img[1], img[2], img[3], img[4]))
            fid.write(struct.pack("<ddd", img[5], img[6], img[7]))
            fid.write(struct.pack("<I", img[8]))
            fid.write(img[9].encode("utf-8") + b"\x00")
            fid.write(struct.pack("<Q", img[10]))
            fid.write(struct.pack("<" + "ddq" * img[10], *img[11]))

    print("Filtering complete!")


if __name__ == '__main__':
    # 1. Define paths
    # ===== change it for another images =====
    sparse_folder = "./data/bicycle/sparse/0"
    images_folder = "./data/bicycle/images"
    
    input_images_bin = os.path.join(sparse_folder, "images_original.bin")
    output_images_bin = os.path.join(sparse_folder, "images.bin")
    
    # Backup the original images.bin before overwriting
    original_bin = os.path.join(sparse_folder, "images.bin")
    if os.path.exists(original_bin) and not os.path.exists(input_images_bin):
        os.rename(original_bin, input_images_bin)
        print(f"Backed up original images.bin to images_original.bin")
    
    # 2. Get the exact names of the 4 images you placed in the images folder
    valid_names = set()
    for img_path in glob.glob(os.path.join(images_folder, "*")):
        valid_names.add(os.path.basename(img_path))
        
    print(f"Found {len(valid_names)} target images in the images folder.")
    
    # 3. Run the filter
    filter_images_bin(input_images_bin, output_images_bin, valid_names)
