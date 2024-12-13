import os
import shutil
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import argparse


def arg_parser():
    parser = argparse.ArgumentParser(
        description="Split dataset into train, val, and test subsets."
    )
    parser.add_argument(
        "--image_dir", type=str, help="Path to the directory containing .jpg image files."
    )
    parser.add_argument(
        "--annotation_dir",
        type=str,
        help="Path to the directory containing .png annotation files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Path to the output directory for train/val/test splits.",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Proportion of dataset for testing. Default: 0.2",
    )
    parser.add_argument(
        "--val_size",
        type=float,
        default=0.2,
        help="Proportion of training data for validation. Default: 0.2",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed for reproducibility. Default: 42",
    )
    return parser


def train_test_val_split(
    image_dir,
    annotation_dir,
    output_dir,
    test_size=0.2,
    val_size=0.2,
    random_seed=42,
):
    """
    Perform a train-test-validation split for paired image and annotation files.

    Parameters:
    - image_dir: Directory containing `.jpg` files.
    - annotation_dir: Directory containing `.png` files.
    - output_dir: Base directory for train/val/test splits.
    - test_size: Proportion of the dataset to include in the test split.
    - val_size: Proportion of the remaining data (after test split) to use for validation.
    - random_seed: Seed for reproducibility.
    """
    # Define output directories
    train_image_dir = os.path.join(output_dir, "images/train")
    val_image_dir = os.path.join(output_dir, "images/val")
    test_image_dir = os.path.join(output_dir, "images/test")

    train_annotation_dir = os.path.join(output_dir, "masks/train")
    val_annotation_dir = os.path.join(output_dir, "masks/val")
    test_annotation_dir = os.path.join(output_dir, "masks/test")

    for d in [
        train_image_dir,
        val_image_dir,
        test_image_dir,
        train_annotation_dir,
        val_annotation_dir,
        test_annotation_dir,
    ]:
        os.makedirs(d, exist_ok=True)

    # Get all image files and annotation files
    image_files = {
        f.split(".")[0]: f for f in os.listdir(image_dir) if f.endswith(".jpg")
    }
    annotation_files = {
        f.split(".")[0]: f for f in os.listdir(annotation_dir) if f.endswith(".png")
    }

    # Match files with annotations
    common_keys = set(image_files.keys()) & set(annotation_files.keys())

    # Create full paths for matched files
    matched_image_paths = [
        os.path.join(image_dir, image_files[key]) for key in common_keys
    ]
    matched_annotation_paths = [
        os.path.join(annotation_dir, annotation_files[key]) for key in common_keys
    ]

    # Split into test and train+val sets
    train_val_images, test_images, train_val_annotations, test_annotations = (
        train_test_split(
            matched_image_paths,
            matched_annotation_paths,
            test_size=test_size,
            random_state=random_seed,
        )
    )

    # Split train+val into train and validation sets
    train_images, val_images, train_annotations, val_annotations = train_test_split(
        train_val_images,
        train_val_annotations,
        test_size=val_size,
        random_state=random_seed,
    )

    # Define helper function to copy files
    def copy_files(file_list, target_dir):
        for file_path in tqdm(file_list, desc=f"Copying files to {target_dir}"):
            shutil.copy(
                file_path, os.path.join(target_dir, os.path.basename(file_path))
            )

    # Copy files to respective directories
    copy_files(train_images, train_image_dir)
    copy_files(val_images, val_image_dir)
    copy_files(test_images, test_image_dir)
    copy_files(train_annotations, train_annotation_dir)
    copy_files(val_annotations, val_annotation_dir)
    copy_files(test_annotations, test_annotation_dir)

    # Print summary
    print(f"Train: {len(train_images)} files")
    print(f"Validation: {len(val_images)} files")
    print(f"Test: {len(test_images)} files")


if __name__ == "__main__":
    # Argument parser setup
    args = arg_parser().parse_args()

    # Call the function with parsed arguments
    train_test_val_split(
        image_dir=args.image_dir,
        annotation_dir=args.annotation_dir,
        output_dir=args.output_dir,
        test_size=args.test_size,
        val_size=args.val_size,
        random_seed=args.random_seed,
    )
