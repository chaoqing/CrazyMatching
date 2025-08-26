import cv2
import numpy as np
import os
from pathlib import Path
import random
import json
from tqdm import tqdm

def _extract_animal_images_auto(image: np.ndarray, min_area=500):
    """
    Automatically detects and extracts colored objects from an image using
    Otsu's thresholding to robustly handle lighting variations.
    Includes debugging steps.

    Args:
        min_area (int): The minimum area of a component to be saved.
    """
    all_outputs = {}

    # --- 1. Robust Background Segmentation using Otsu's Thresholding ---
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Otsu's binarization
    # This automatically determines the best threshold value.
    threshold_value, otsu_mask = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    print(f"Otsu's algorithm determined the optimal threshold to be: {threshold_value}")

    # Invert the mask. In the original Otsu mask, the background is likely white
    # (255) and the objects are black (0). We want the opposite for component analysis.
    object_mask = cv2.bitwise_not(otsu_mask)

    # --- Card and Background Extraction ---
    print("\nAttempting to extract cards and background...")

    # Now that we have solid cards, we can find them as components.
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        otsu_mask, connectivity=8
    )

    # We expect 3 components: background (label 0) and two cards.
    if num_labels >= 3:
        # Get the areas of the components, excluding the background.
        areas = stats[1:, cv2.CC_STAT_AREA]
        # Sort by area to find the two largest.
        sorted_component_indices = np.argsort(areas)[::-1]

        if len(sorted_component_indices) >= 2:
            print("Found two largest components, treating them as cards.")
            # Get the original labels of the two largest components
            card1_label = sorted_component_indices[0] + 1
            card2_label = sorted_component_indices[1] + 1

            # Create separate masks for each card.
            card1_mask = np.uint8(labels == card1_label) * 255
            card2_mask = np.uint8(labels == card2_label) * 255

            # First, fill the holes (the animals) in the card regions.
            # The object_mask has white cards with holes on a black background.
            contours, _ = cv2.findContours(
                card1_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            card1_mask_filled = np.zeros_like(card1_mask)
            cv2.drawContours(
                card1_mask_filled, contours, -1, (255), thickness=cv2.FILLED
            )

            contours, _ = cv2.findContours(
                card2_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            card2_mask_filled = np.zeros_like(card2_mask)
            cv2.drawContours(
                card2_mask_filled, contours, -1, (255), thickness=cv2.FILLED
            )

            # The background is everything that is not a card.
            # We can get this by inverting the filled_cards_mask.
            background_mask = cv2.bitwise_not(card1_mask_filled | card2_mask_filled)

            # Create a 4-channel image (BGRA) using the solid mask
            card1_image = cv2.merge([*cv2.split(image), card1_mask])
            card2_image = cv2.merge([*cv2.split(image), card2_mask])

            # --- Create Background Image ---
            # Using INPAINT_NS (Navier-Stokes based) can produce different, sometimes better, results
            # for removing large objects compared to INPAINT_TELEA. A larger radius is also used.
            print("\nInpainting background to remove cards...")
            background_image = cv2.inpaint(
                image, cv2.bitwise_not(background_mask), 3, cv2.INPAINT_NS
            )
            print("Background inpainting complete.")

            all_outputs.update({
                "card1": card1_image,
                "card2": card2_image,
                "background": background_image,
            })
        else:
            print("Did not find two large enough components to be considered cards.")
    else:
        print("Not enough components found to extract cards (less than 3).")

    # DEBUG: Save the object mask
    all_outputs.update({
        "debug_object_mask": object_mask,
    })

    # --- 2. Connected Component Analysis ---
    print("\nStarting connected component analysis...")
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        object_mask, 4, cv2.CV_32S
    )
    print(f"Found {num_labels - 1} potential components (excluding background).")

    # DEBUG: Create an image to visualize the labeled components
    label_hue = np.uint8(179 * labels / np.max(labels))
    blank_ch = 255 * np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
    labeled_img[label_hue == 0] = 0  # Set background to black

    all_outputs.update({
        "debug_labeled_components": labeled_img,
    })

    # --- 3. Statistical Analysis to Find Outliers ---
    print("\nPerforming statistical analysis on component areas...")
    component_areas = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > 0:
            component_areas.append(area)

    if len(component_areas) > 2:
        mean_area = np.mean(component_areas)
        std_dev_area = np.std(component_areas)
        # Define a component as an outlier if it's more than 2 std deviations from the mean
        lower_bound = mean_area - 2 * std_dev_area
        upper_bound = mean_area + 2 * std_dev_area
        print(f"Area stats: Mean={mean_area:.2f}, StdDev={std_dev_area:.2f}")
        print(
            f"Valid area range (mean +/- 2*std): ({lower_bound:.2f}, {upper_bound:.2f})"
        )
    else:
        # Not enough data for meaningful stats, so we won't treat any as outliers
        print(
            "Not enough components for statistical analysis. Skipping outlier detection."
        )
        lower_bound, upper_bound = -1, float("inf")

    # --- 4. Filtering and Extraction ---
    debug_bbox_image = image.copy()
    extracted_count = 0
    outlier_count = 0
    # Skip the first component (label 0), as it's the background
    for i in range(1, num_labels):
        # Get component properties
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        # First, filter out components that are too small to even be considered
        if area < min_area:
            continue

        print(f"\n--- Processing Component {i} ---")
        print(f"Area: {area}")
        print(f"Bounding Box: (x={x}, y={y}, w={w}, h={h})")

        # Check if the component is an outlier
        is_outlier = not (lower_bound < area < upper_bound)

        if is_outlier:
            outlier_count += 1
            file_prefix = "outlier_animal_"
            print(f"Result: Flagged as OUTLIER.")
        else:
            extracted_count += 1
            file_prefix = "animal_"
            print(f"Result: Identified as valid component.")

        # Draw bounding box on the debug image
        color = (0, 0, 255) if is_outlier else (0, 255, 0)
        cv2.rectangle(debug_bbox_image, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            debug_bbox_image,
            str(i),
            (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )

        # Crop the object from the original image
        cropped_image = image[y : y + h, x : x + w]

        # Create a transparent background by filling the component's contour
        component_mask = (labels[y : y + h, x : x + w] == i).astype(np.uint8) * 255

        # Use a morphological closing operation to fill small holes and smooth the shape
        kernel = np.ones((5, 5), np.uint8)
        closed_mask = cv2.morphologyEx(component_mask, cv2.MORPH_CLOSE, kernel)

        # Find the external contour of the component to fill any holes
        contours, _ = cv2.findContours(
            closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Create a new solid mask by drawing the filled contour
        solid_mask = np.zeros_like(component_mask)
        cv2.drawContours(solid_mask, contours, -1, (255), cv2.FILLED)

        # Create a 4-channel image (BGRA) using the solid mask
        b, g, r = cv2.split(cropped_image)
        transparent_image = cv2.merge([b, g, r, solid_mask])

        # Save the extracted image with a transparent background

        all_outputs.update({
            f"{file_prefix}{i}": transparent_image,
        })
    print(
        f"\nExtraction complete. Found {extracted_count} valid components and {outlier_count} outliers."
    )

    # DEBUG: Save the image with all bounding boxes
    all_outputs.update({
        f"debug_bounding_boxes": debug_bbox_image,
    })
    return all_outputs

def extract_animal_images_auto(image_path, output_dir, min_area=500):
    """
    Automatically detects and extracts colored objects from an image using
    Otsu's thresholding to robustly handle lighting variations.
    Includes debugging steps.

    Args:
        image_path (str): Path to the source image.
        output_dir (str): Directory to save the extracted and debug images.
        min_area (int): The minimum area of a component to be saved.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(f"Created output directory: {output_dir}")

    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return
    print(f"Successfully loaded image: {image_path}")

    for k, v in _extract_animal_images_auto(image, min_area=min_area).items():
        image_path = os.path.join(output_dir, f"{k}.png")
        cv2.imwrite(image_path, v)
        print(f"Saved debug bounding boxes image to: {image_path}")


if __name__ == "__main__":
    image_file = Path(__file__).with_name("..")/'../.gemini/IMG_20250720_090212_edit_730440424190105.jpg'
    output_directory_extract = Path(__file__).with_name('extracted_animals')
    extract_animal_images_auto(image_file, output_directory_extract, min_area=1000)

    for p in (Path(__file__).with_name("..")/"../debug-logs/").glob("*.json"):
        output_dir = p.with_suffix("")
        output_dir.mkdir(exist_ok=True)

        data = json.loads(p.read_text())
        img = np.array(data["img"]["data"], dtype=np.uint8).reshape(*data["img"]["shape"])
        img = img[:,:, ::-1]
        cv2.imwrite(str(output_dir/"input.png"), img)

        for k, v in _extract_animal_images_auto(img, min_area=500).items():
            image_path = os.path.join(output_dir, f"{k}.png")
            cv2.imwrite(image_path, v)
            print(f"Saved debug bounding boxes image to: {image_path}")
