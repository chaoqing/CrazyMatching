import cv2
import numpy as np
import os
from pathlib import Path
import random
import json
from tqdm import tqdm


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

            # Save the individual card and the background image.
            card1_path = os.path.join(output_dir, "card1.png")
            cv2.imwrite(card1_path, card1_image)
            print(f"Saved card 1 image to: {card1_path}")

            card2_path = os.path.join(output_dir, "card2.png")
            cv2.imwrite(card2_path, card2_image)
            print(f"Saved card 2 image to: {card2_path}")

            background_path = os.path.join(output_dir, "background.png")
            cv2.imwrite(background_path, background_image)
            print(f"Saved background image to: {background_path}")
        else:
            print("Did not find two large enough components to be considered cards.")
    else:
        print("Not enough components found to extract cards (less than 3).")

    # DEBUG: Save the object mask
    debug_mask_path = os.path.join(output_dir, "debug_object_mask.png")
    cv2.imwrite(debug_mask_path, object_mask)
    print(f"Saved debug mask (from Otsu) to: {debug_mask_path}")

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

    debug_labeled_path = os.path.join(output_dir, "debug_labeled_components.png")
    cv2.imwrite(debug_labeled_path, labeled_img)
    print(f"Saved debug labeled components image to: {debug_labeled_path}")

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
        output_path = os.path.join(output_dir, f"{file_prefix}{i}.png")
        cv2.imwrite(output_path, transparent_image)
        print(f"Saved image to: {output_path}")

    print(
        f"\nExtraction complete. Found {extracted_count} valid components and {outlier_count} outliers."
    )

    # DEBUG: Save the image with all bounding boxes
    debug_bbox_path = os.path.join(output_dir, "debug_bounding_boxes.png")
    cv2.imwrite(debug_bbox_path, debug_bbox_image)
    print(f"Saved debug bounding boxes image to: {debug_bbox_path}")


def overlay_image(background, overlay, x, y):
    """
    Overlays a transparent RGBA image onto a BGR background image at a specific location.
    """
    # Ensure overlay is 4 channels
    if overlay.shape[2] == 3:
        print("Warning: Overlay image is not transparent. Converting to RGBA.")
        overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2BGRA)

    h, w, _ = overlay.shape
    b_h, b_w, _ = background.shape

    # Top-left corner of the paste
    x1, y1 = x, y
    # Bottom-right corner of the paste
    x2, y2 = x + w, y + h

    # The part of the overlay that is inside the background
    overlay_x1, overlay_y1 = 0, 0
    overlay_x2, overlay_y2 = w, h

    # Clip paste coordinates to background dimensions
    if x1 < 0:
        overlay_x1 = -x1
        x1 = 0
    if y1 < 0:
        overlay_y1 = -y1
        y1 = 0
    if x2 > b_w:
        overlay_x2 -= x2 - b_w
        x2 = b_w
    if y2 > b_h:
        overlay_y2 -= y2 - b_h
        y2 = b_h

    # Get the region of interest on the background
    roi = background[y1:y2, x1:x2]

    # Crop the overlay image to match the clipped ROI
    overlay_cropped = overlay[overlay_y1:overlay_y2, overlay_x1:overlay_x2]

    # Ensure the ROI and overlay have the same dimensions
    if (
        roi.shape[0] != overlay_cropped.shape[0]
        or roi.shape[1] != overlay_cropped.shape[1]
    ):
        # This can happen if the paste is completely off-screen
        return background

    # Split overlay into color and alpha channels
    b, g, r, a = cv2.split(overlay_cropped)
    overlay_rgb = cv2.merge((b, g, r))

    # Normalize alpha mask to be between 0 and 1
    alpha = a / 255.0
    alpha = np.expand_dims(alpha, axis=2)  # for broadcasting

    # Perform alpha blending
    blended_roi = (alpha * overlay_rgb + (1 - alpha) * roi).astype(np.uint8)

    background[y1:y2, x1:x2] = blended_roi
    return background


def scale_and_rotate(img: np.ndarray, scale_factor: float, angle: float):
    h, w, *_ = img.shape
    new_w, new_h = int(w * scale_factor), int(h * scale_factor)
    img_scaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    h, w = img_scaled.shape[:2]
    center = (w // 2, h // 2)

    m = cv2.getRotationMatrix2D(center, np.degrees(-angle), 1.0)

    cos = np.abs(m[0, 0])
    sin = np.abs(m[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    m[0, 2] += (new_w / 2) - center[0]
    m[1, 2] += (new_h / 2) - center[1]

    img_rotated = cv2.warpAffine(
        img_scaled, m, (new_w, new_h), borderValue=(0, 0, 0, 0)
    )

    coords = cv2.findNonZero(img_rotated[:, :, 3])
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        return img_rotated[y : y + h, x : x + w]
    else:
        return img_rotated


def generate_training_data(num_samples=100):
    """
    Generates simulated training data for the Crazy Matching game.
    This is a generator that yields one raw sample at a time.
    """
    data_dir = Path(__file__).parent

    background_path = data_dir / "background.png"
    assert background_path.exists(), (
        f"Error: Background image not found at {background_path}"
    )
    background_img = cv2.imread(str(background_path))
    assert background_img is not None, (
        f"Error: Could not read background image from {background_path}"
    )
    bg_h, bg_w, _ = background_img.shape

    animal_imgs = {}
    for p in data_dir.glob("animal_*.png"):
        animal_img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        assert animal_img is not None, f"Error: Can not read {p}"
        assert animal_img.shape[2] == 4, f"Error: {p} does not have alpha channel"
        coords = cv2.findNonZero(animal_img[:, :, 3])
        assert coords is not None, f"Error: {p} does not have colored pixels"
        x, y, w, h = cv2.boundingRect(coords)
        animal_imgs[p.name.rstrip(".png")] = animal_img[y : y + h, x : x + w]
    assert animal_imgs is not [], (
        "Error: No animal images found in 'data'. Run the extraction first."
    )

    animal_names = tuple(animal_imgs.keys())
    for i in range(num_samples):
        selected_animal_paths = random.sample(animal_names, 9)
        duplicated_animal_path = selected_animal_paths[0]

        animal_patchs = []
        for animal in selected_animal_paths + [duplicated_animal_path]:
            scale = random.uniform(0.5, 1.0)
            rotation = random.uniform(-np.pi, np.pi)

            animal_patchs.append(
                (
                    scale_and_rotate(animal_imgs[animal], scale, rotation),
                    animal,
                    scale,
                    rotation,
                )
            )

        # Create a fresh copy of the background for each sample
        sample_image = background_img.copy()

        rectangles = []
        attempts = 0
        while len(animal_patchs) > 0 and attempts < 500:
            attempts += 1

            # Use a fixed size for simplicity, scaled by the random factor
            rect_h, rect_w = animal_patchs[-1][0].shape[:2]

            # Randomly position the center, ensuring it's not too close to the edge
            center_x = random.randint(rect_w, bg_w - rect_w)
            center_y = random.randint(rect_h, bg_h - rect_h)

            # Simple collision detection
            is_overlapping = False
            for r in rectangles:
                location = r["location"]
                dist = np.sqrt(
                    (center_x - location["center_x"]) ** 2
                    + (center_y - location["center_y"]) ** 2
                )
                if dist < (rect_w + location["w"]) / 1.5:  # Looser check
                    is_overlapping = True
                    break

            if not is_overlapping:
                animal_img, animal, scale, rotation = animal_patchs.pop()
                overlay_image(
                    sample_image,
                    animal_img,
                    center_x - rect_w // 2,
                    center_y - rect_h // 2,
                )

                rectangles.append(
                    {
                        "location": {
                            "center_x": center_x,
                            "center_y": center_y,
                            "w": rect_w,
                            "h": rect_h,
                        },
                        "transform": {"scale": scale, "rotation": rotation},
                        "label": animal,
                    }
                )

        if len(rectangles) < 10:
            print(
                f"Warning: Could only place {len(rectangles)} non-overlapping rectangles for sample {i}."
            )
            continue

        yield sample_image, rectangles


def create_dataset(
    output_dir, num_samples=100, image_size=(320, 320), save_raw=False, only_pair=True
):
    """
    Creates a dataset by calling the data generator, processing the data,
    and saving it to disk.

    - Saves raw generated images to a subfolder.
    - Resizes images and collects them into a NumPy array.
    - Collects labels into a NumPy array.
    - Saves the final NumPy arrays for training.
    """
    images_dir = Path(output_dir) / "images"
    labels_dir = Path(output_dir) / "labels"
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    images_batch = []
    location_batch = []
    labels_batch = []

    data_generator = tqdm(
        generate_training_data(num_samples=num_samples),
        total=num_samples,
        desc=f"Dataset samples",
    )

    for i, (image, label) in enumerate(data_generator):
        if save_raw:
            image_filename = f"sample_{i:04d}.png"
            label_filename = f"sample_{i:04d}.json"

            cv2.imwrite(str(images_dir / image_filename), image)
            with open(labels_dir / label_filename, "w") as f:
                json.dump(label, f, indent=2)

        # Resize image for the training batch
        resized_image = cv2.resize(image, image_size)
        images_batch.append(resized_image)

        # Append the label

        if only_pair:
            label = label[::9]

        locations = np.array(
            [list(i["location"].values()) for i in label], dtype=np.float32
        )
        locations[:, 0::2] *= image_size[1] / image.shape[1]
        locations[:, 1::2] *= image_size[0] / image.shape[0]

        location_batch.append(locations)
        labels_batch.append([i["label"] for i in label])

    # Convert lists to NumPy arrays
    X_train = np.array(images_batch)
    y_train = np.array(location_batch)
    c_train = np.array(labels_batch)

    # Normalize image data
    X_train = X_train.astype("float32")
    y_train = y_train.astype("float32")

    # Save the NumPy arrays
    X_train_path = output_dir / "X_train.npy"
    y_train_path = output_dir / "y_train.npy"
    c_train_path = output_dir / "c_train.npy"

    np.save(X_train_path, X_train)
    np.save(y_train_path, y_train)
    np.save(c_train_path, c_train)

    print("\nDataset creation complete.")
    print(f"  - Shape of image data (X_train): {X_train.shape}")
    print(f"  - Shape of location data (y_train): {y_train.shape}")
    print(f"  - Shape of label data (c_train): {c_train.shape}")
    print(f"Original images saved in: '{images_dir}'")
    print(f"Training data saved as '{X_train_path}' and '{y_train_path}'")


def debug_training_data(image_path, label_path, output_path):
    """
    Reads a training sample and its label, and draws the bounding boxes on the image
    to visually verify the data. Also saves a resized version for inspection.
    """

    if image_path.name.endswith(".npy"):
        # If the image is a NumPy array, load it
        images = np.load(image_path)
        ith_image = np.random.choice(
            range(images.shape[0])
        )  # Randomly select one image
        image = images[ith_image]

        label_data = [
            {
                "location": {
                    "center_x": i[0],
                    "center_y": i[1],
                    "w": i[2],
                    "h": i[3],
                },
                "transform": {"rotation": 0.0},
            }
            for i in np.load(label_path)[ith_image]
        ]
    else:
        # Load the image
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Error: Could not load image from {image_path}")
            return

        # Load the label
        with open(label_path, "r") as f:
            label_data = json.load(f)

    debug_image = image.copy()

    for i, label in enumerate(label_data):
        box_data = label["location"]
        # Denormalize the coordinates
        center_x = box_data["center_x"]
        center_y = box_data["center_y"]
        w = box_data["w"]
        h = box_data["h"]

        rotation_rad = label["transform"]["rotation"]
        rotation_deg = 0  # np.degrees(rotation_rad)

        # Get the 4 corners of the rotated rectangle
        box_points = cv2.boxPoints(((center_x, center_y), (w, h), rotation_deg))
        box_points = np.intp(box_points)

        # Draw the rectangle
        color = (
            (0, 255, 0) if i == 0 else (0, 0, 255)
        )  # Green for first, Red for second
        cv2.drawContours(debug_image, [box_points], 0, color, 2)

    # Save the original-sized debug image
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(str(output_path), debug_image)
    print(f"Saved debug image with bounding boxes to: {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate simulated training data for TFOD."
    )
    parser.add_argument(
        "--num_samples", type=int, default=100, help="Number of samples to generate."
    )
    parser.add_argument(
        "--save_raw",
        action="store_true",
        default=False,
        help="Also save the raw image before resizing.",
    )
    parser.add_argument(
        "--only_pair",
        action="store_true",
        default=False,
        help="The output will be only the first and last same pair animals.",
    )
    parser.add_argument(
        "--output_base_dir",
        type=str,
        default=str(Path(__file__).parent / "training_data"),
        help="Base directory for output images, annotations, and debug_output.",
    )
    args = parser.parse_args()

    # --- IMPORTANT ---
    # Ensure you have opencv-python installed:
    # pip install opencv-python numpy
    # -----------------

    # --- Part 1: Extract animal assets from a source image ---
    # This part is for initial asset creation.
    # Uncomment and run this first if you don't have the animal images.
    # print("--- Running Step 1: Extracting Animal Images ---")
    # image_file = '.gemini/IMG_20250720_090212_edit_730440424190105.jpg'
    # output_directory_extract = Path(__file__).parent/'extracted_animals'
    # extract_animal_images_auto(image_file, output_directory_extract, min_area=1000)
    # print("\n--- Finished Step 1 ---\n")

    # --- Part 2: Generate simulated training data ---
    # This part uses the extracted assets to create training samples.
    print("--- Running Step 2: Generating Training Data ---")
    output_directory_sim = Path(args.output_base_dir)
    create_dataset(
        output_directory_sim,
        num_samples=args.num_samples,
        save_raw=args.save_raw,
        only_pair=args.only_pair,
    )  # Generate 50 samples for testing
    print("\n--- Finished Step 2 ---")

    # --- Part 3: Debug and verify a training sample ---
    print("\n--- Running Step 3: Debugging a Training Sample ---")
    sim_dir = output_directory_sim
    image_files = list((sim_dir / "images").glob("*.png"))

    if not image_files:
        for i in range(5):
            debug_output_path = (
                Path(__file__).parent / "debug_output" / f"sample_{i:02}.png"
            )
            debug_training_data(
                sim_dir / "X_train.npy", sim_dir / "y_train.npy", debug_output_path
            )
    else:
        # Pick a random image to debug
        random_image_path = random.choice(image_files)
        label_path = sim_dir / "labels" / (random_image_path.stem + ".json")
        debug_output_path = (
            Path(__file__).parent / "debug_output" / random_image_path.name
        )

        if label_path.exists():
            debug_training_data(random_image_path, label_path, debug_output_path)
        else:
            print(
                f"Error: Could not find corresponding label for {random_image_path.name}"
            )
    print("\n--- Finished Step 3 ---")
