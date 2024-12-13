import argparse
import os
import shutil
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


def copy_and_rename_files(source_dir, target_dir):
    # Ensure the target directory exists
    os.makedirs(target_dir, exist_ok=True)

    # Collect all files with the specified suffix
    files_to_process = []
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.endswith("_gtFine_labelIds.png"):
                files_to_process.append((root, file))

    # Iterate through files with progress bar
    for root, file in tqdm(files_to_process, desc="Processing files", unit="file"):
        # Construct the full source file path
        source_path = os.path.join(root, file)

        # Create the new file name by replacing the suffix
        new_name = file.replace("_gtFine_labelIds.png", ".png")

        # Construct the target file path
        target_path = os.path.join(target_dir, new_name)

        # Copy the file to the target directory with the new name
        shutil.copy2(source_path, target_path)

    print("Files have been copied and renamed.")


if __name__ == "__main__":
    args = parse_arguments()
    copy_and_rename_files(args["source"], args["target"])
