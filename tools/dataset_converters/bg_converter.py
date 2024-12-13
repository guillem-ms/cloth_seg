import argparse
import os
import numpy as np
from PIL import Image
from tqdm import tqdm


def parse_arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-s", "--source", required=True, help="Path to the source directory."
    )
    ap.add_argument(
        "-t", "--target", required=True, help="Path to the target directory."
    )
    return vars(ap.parse_args())


def process_masks(input_dir, output_dir):
    """
    Process .png mask files in the input directory and map pixel values:
    - Map all non-zero values to 1, except for 30, which is mapped to 0.

    Parameters:
    - input_dir: str, path to the input directory containing .png masks.
    - output_dir: str, path to the output directory where processed masks will be saved.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for file_name in tqdm(os.listdir(input_dir)):
        if file_name.endswith(".png"):
            # Load the mask image
            mask_path = os.path.join(input_dir, file_name)
            mask = np.array(Image.open(mask_path))

            # Apply the transformation
            processed_mask = np.where(mask == 31, 0, np.where(mask > 0, 1, 0))

            # Save the processed mask
            output_path = os.path.join(output_dir, file_name)
            Image.fromarray(processed_mask.astype(np.uint8)).save(output_path)


if __name__ == "__main__":
    args = parse_arguments()
    process_masks(args["source"], args["target"])
