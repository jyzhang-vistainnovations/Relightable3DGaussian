import os
import torch
import numpy as np
from PIL import Image


def convert_depth_to_mask(input_folder, output_folder):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate through all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".pt"):
            # Load the depth map tensor
            depth_map_path = os.path.join(input_folder, filename)
            depth_tensor = torch.load(depth_map_path)

            # Move tensor to CPU if it's on CUDA
            if depth_tensor.is_cuda:
                depth_tensor = depth_tensor.cpu()

            # Convert to numpy array
            depth_array = depth_tensor.numpy()

            # Ensure we're working with a 2D array
            if depth_array.ndim > 2:
                # If it's a 3D array, take the first channel or squeeze
                depth_array = depth_array.squeeze()
                if depth_array.ndim > 2:
                    depth_array = depth_array[0]  # Take the first channel if still 3D

            # Create binary mask (1 where depth is non-zero, 0 otherwise)
            mask = np.where(depth_array != 0, 255, 0).astype(np.uint8)

            # Create a new image from the mask
            mask_image = Image.fromarray(mask)

            # Save the mask as PNG
            output_filename = os.path.splitext(filename)[0] + ".png"
            output_path = os.path.join(output_folder, output_filename)
            mask_image.save(output_path)

            print(f"Processed: {filename} -> {output_filename}")


# Usage
input_folder = "c://Users/U/Documents/gs/RaDe-GS/outputs/model-3-1600/full_depth_data"
output_folder = "c://Users/U/Documents/gs/Relightable/inputs/model-3-1600/mask"

convert_depth_to_mask(input_folder, output_folder)
