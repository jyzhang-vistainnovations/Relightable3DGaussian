import os
import numpy as np
from PIL import Image


def apply_masks_to_images(image_folder, mask_folder, output_folder):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get list of files in image and mask folders
    image_files = sorted(os.listdir(image_folder))
    mask_files = sorted(os.listdir(mask_folder))

    # Ensure we have the same number of images and masks
    if len(image_files) != len(mask_files):
        print("Error: Number of images and masks do not match.")
        return

    for image_file, mask_file in zip(image_files, mask_files):
        # Load image
        image_path = os.path.join(image_folder, image_file)
        image = Image.open(image_path).convert("RGBA")

        # Load mask
        mask_path = os.path.join(mask_folder, mask_file)
        mask = Image.open(mask_path).convert("L")

        # Resize mask if it doesn't match the image size
        if mask.size != image.size:
            print(f"Resizing mask for {image_file}")
            mask = mask.resize(image.size, Image.LANCZOS)

        # Convert images to numpy arrays
        image_array = np.array(image)
        mask_array = np.array(mask)

        # Create an alpha channel from the mask
        alpha_channel = mask_array

        # Apply the alpha channel to the image
        image_array[:, :, 3] = alpha_channel

        # Create a new image with transparency
        masked_image = Image.fromarray(image_array, "RGBA")

        # Save the masked image as PNG
        output_filename = f"{os.path.splitext(image_file)[0]}.png"
        output_path = os.path.join(output_folder, output_filename)
        masked_image.save(output_path, format="PNG")

        print(f"Processed: {image_file} with {mask_file} -> {output_filename}")


# Your folder paths remain the same
image_folder = "c://Users/U/Documents/gs/captures/model-3-1600/images"
mask_folder = "c://Users/U/Documents/gs/Relightable/inputs/model-3-1600/mask"
output_folder = "c://Users/U/Documents/gs/Relightable/inputs/model-3-1600/images"

apply_masks_to_images(image_folder, mask_folder, output_folder)
