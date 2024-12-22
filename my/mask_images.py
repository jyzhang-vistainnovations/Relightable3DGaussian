import os
import numpy as np
from PIL import Image
import argparse


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
        # Convert to RGB instead of RGBA since JPG doesn't support transparency
        image = Image.open(image_path).convert("RGB")

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

        # Apply mask to RGB channels
        # Create a boolean mask where mask is non-zero
        mask_bool = mask_array > 0
        # Broadcast the mask to all channels
        mask_3d = np.stack([mask_bool] * 3, axis=2)
        # Set background (where mask is zero) to white
        image_array[~mask_3d] = 255

        # Create a new image
        masked_image = Image.fromarray(image_array, "RGB")

        # Save the masked image as JPG with maximum quality
        output_filename = f"{os.path.splitext(image_file)[0]}.jpg"
        output_path = os.path.join(output_folder, output_filename)
        masked_image.save(output_path, format="JPEG", quality=95, optimize=False)

        print(f"Processed: {image_file} with {mask_file} -> {output_filename}")


def main():
    parser = argparse.ArgumentParser(
        description="Apply masks to images and save as high-quality JPG"
    )
    parser.add_argument(
        "image_folder", help="Path to the folder containing input images"
    )
    parser.add_argument("mask_folder", help="Path to the folder containing mask images")
    parser.add_argument(
        "output_folder", help="Path to the folder where masked images will be saved"
    )

    args = parser.parse_args()

    apply_masks_to_images(args.image_folder, args.mask_folder, args.output_folder)


if __name__ == "__main__":
    main()
