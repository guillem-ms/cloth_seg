import json
import os
import cv2
import numpy as np
from tqdm import tqdm


def save_json(data: dict, file_path: str):
    """
    Save a dictionary to a JSON file.
    :param data: The dictionary to save.
    :param output_path: The path to the output JSON file.
    """
    with open(file_path, "w") as f:
        json.dump(data, f)


def sem_seg_tensor_output_to_instance_dict(
    original_image_path: str,
    binary_mask: np.ndarray,
) -> dict | None:
    """
    Given a binary mask, containing id per pixel, produce a dictionary describing the largest instance.

    1. Perform mask refinement using Graph Cuts with the original image and the mask.
    2. Detect the largest blob and extract it as a single instance.
    3. Compute the polygon coordinates (x, y) and bounding box of the largest blob.
    4. Add confidence and label fields.

    :param original_image_path: str
        Path to the original image.
    :param binary_mask: numpy.ndarray
        Binary mask - (H x W) uint8 array.
    :return: dict | None
        Dictionary containing instance data or None if no valid instance is found.
    """
    # Load the original image
    original_image = cv2.imread(original_image_path)
    if original_image is None:
        raise FileNotFoundError(f"Image not found: {original_image_path}")

    # Ensure binary_mask is binary (0 or 1)
    binary_mask = (binary_mask > 0).astype(np.uint8)

    # save mask here
    cv2.imwrite("bin_mask.png", binary_mask[0] * 255)

    # Prepare the mask for Graph Cuts
    mask_fg = binary_mask.copy()[0] * 255
    mask_bg = cv2.bitwise_not(mask_fg)
    # Define the foreground and background models for Graph Cuts
    mask = np.zeros(original_image.shape[:2], np.uint8)  # Initialize mask for GrabCut
    mask[binary_mask[0] == 1] = cv2.GC_PR_FGD
    mask[binary_mask[0] == 0] = cv2.GC_PR_BGD
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    # Apply Graph Cuts
    refined_mask = mask.copy()
    cv2.grabCut(
        original_image,
        refined_mask,
        None,
        bgd_model,
        fgd_model,
        5,
        cv2.GC_INIT_WITH_MASK,
    )
    refined_mask = np.where(
        (refined_mask == cv2.GC_FGD) | (refined_mask == cv2.GC_PR_FGD), 1, 0
    ).astype(np.uint8)

    cv2.imwrite("bin_mask_ref.png", refined_mask * 255)

    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        refined_mask, connectivity=8
    )

    if num_labels <= 1:  # No valid blobs found
        return None

    # Find the largest blob (ignoring the background, label 0)
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    largest_mask = (labels == largest_label).astype(np.uint8)

    # Find contours for the largest blob
    contours, _ = cv2.findContours(
        largest_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None

    # Take the largest contour
    largest_contour = max(contours, key=cv2.contourArea)

    # Convert contour points to (x, y) pairs
    polygon = [(int(point[0][0]), int(point[0][1])) for point in largest_contour]

    # Compute the bounding box
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Create the result dictionary
    instance_dict = {
        "mask": polygon,
        "bbox": [x, y, x + w, y + h],
        "confidence": 1.0,  # Confidence is fixed to 1.0 as per requirements
        "category:_id": 1,  # Fixed label 1 for the single instance
    }

    return [instance_dict]


def seg_results_to_json(
    results,
    save_path: str = "results.json",
    cat_mapping: dict = None,
) -> None:
    """
    Convert YOLO segmentation results to JSON format.

    :param results: YOLO segmentation results.
    :param confidence_threshold: Confidence threshold.
    :param save_path: Path to save the JSON file.
    """
    json_results = {}
    for pred in tqdm(results, desc="Converting to JSON"):
        binary_mask, path = (
            pred.pred_sem_seg.data.cpu().numpy(),
            pred.img_path,
        )
        instances_list_of_dicts = sem_seg_tensor_output_to_instance_dict(
            original_image_path=path, binary_mask=binary_mask
        )
        json_results[os.path.basename(path)] = instances_list_of_dicts

    final_json = {}
    if cat_mapping is not None:
        final_json["id_to_cat_mapping"] = cat_mapping
    final_json["predictions"] = json_results
    save_json(final_json, save_path)
    print(f"Results saved to {save_path}")