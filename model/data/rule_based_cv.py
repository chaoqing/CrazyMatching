import cv2
import numpy as np
import os
from pathlib import Path
import random
import json
from tqdm import tqdm

def _extract_animal_images_auto_advanced(image: np.ndarray, **_):
    """
    Process an image containing exactly 2 cards with white backgrounds and 8 colored animals each.
    Will abort processing if these conditions are not met at any stage.
    
    Args:
        image: BGR format image containing two cards
        _: Unused parameter kept for interface compatibility
        
    Returns:
        Dictionary containing extracted animals and debug images, or empty dict if validation fails
    """
    all_outputs = {}
    
    # Input validation
    if image is None or image.size == 0:
        print("Error: Invalid input image")
        return all_outputs
        
    # Step 1: Enhanced preprocessing for better card detection
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply bilateral filter to reduce noise while preserving edges
    bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Use Canny edge detection to find strong edges
    edges = cv2.Canny(bilateral, 50, 150)
    
    # Dilate edges to connect broken lines
    kernel = np.ones((3,3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=2)
    
    # Apply morphological closing to fill gaps
    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Step 2: Find and filter contours to detect cards
    contours, hierarchy = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Validation: Must find more than 2 contours initially to ensure we're detecting features
    if len(contours) < 2:
        print("Error: Failed to detect sufficient contours in the image")
        return all_outputs
        
    # Add debug output for preprocessing steps
    all_outputs.update({
        "debug_bilateral": bilateral,
        "debug_edges": edges,
        "debug_dilated": dilated,
        "debug_closed": closed
    })
    # Process contours to find cards
    card_contours = []
    min_card_area = image.shape[0] * image.shape[1] * 0.1  # Cards should be at least 10% of image
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_card_area:  # Skip very small contours
            continue
            
        # Approximate the contour to detect rectangles
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        
        # Check if the contour is roughly rectangular (4 corners)
        if len(approx) == 4:
            card_contours.append(cnt)
    
    # Validation: Must find exactly 2 card contours
    if len(card_contours) != 2:
        print(f"Error: Found {len(card_contours)} cards instead of exactly 2")
        return all_outputs
        
    # Sort contours by area
    card_contours = sorted(card_contours, key=cv2.contourArea, reverse=True)
    
    # Validate card areas are similar (within 20% of each other)
    area1 = cv2.contourArea(card_contours[0])
    area2 = cv2.contourArea(card_contours[1])
    area_diff_ratio = abs(area1 - area2) / max(area1, area2)
    
    if area_diff_ratio > 0.2:  # More than 20% difference
        print(f"Error: Card areas differ too much: {area_diff_ratio:.2%}")
        return all_outputs

    # Debug image for card detection
    debug_cards = image.copy()
    cv2.drawContours(debug_cards, card_contours, -1, (0, 255, 0), 2)
    all_outputs["debug_card_detection"] = debug_cards

    # Step 3: Process each card separately
    for idx, card_cnt in enumerate(card_contours, 1):
        # Create mask for the card
        card_mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(card_mask, [card_cnt], -1, (255), -1)
        
        # Extract the card region
        card_region = cv2.bitwise_and(image, image, mask=card_mask)
        
        # Get the bounding rectangle
        x, y, w, h = cv2.boundingRect(card_cnt)
        card_roi = card_region[y:y+h, x:x+w]
        card_mask_roi = card_mask[y:y+h, x:x+w]

        # Convert to HSV for better color segmentation
        hsv = cv2.cvtColor(card_roi, cv2.COLOR_BGR2HSV)
        
        # Create a mask for white background (high value in HSV)
        white_mask = cv2.inRange(hsv, (0, 0, 200), (180, 30, 255))
        
        # Invert to get the animals
        animal_mask = cv2.bitwise_not(white_mask)
        
        # Clean up the mask
        kernel = np.ones((5,5), np.uint8)
        animal_mask = cv2.morphologyEx(animal_mask, cv2.MORPH_OPEN, kernel)
        animal_mask = cv2.morphologyEx(animal_mask, cv2.MORPH_CLOSE, kernel)

        # Find animal components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            animal_mask, connectivity=8
        )
        
        # We expect exactly 8 animals plus background (9 total components)
        if num_labels != 9:
            print(f"Error: Found {num_labels-1} animals in card {idx} instead of 8")
            return all_outputs
            
        # Calculate median area to filter noise
        areas = [stats[i, cv2.CC_STAT_AREA] for i in range(1, num_labels)]
        median_area = np.median(areas)
        
        # Filter components: must be within 40-250% of median area
        valid_components = []
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if 0.4 * median_area <= area <= 2.5 * median_area:
                valid_components.append(i)
                
        # Validate: must have exactly 8 valid animals
        if len(valid_components) != 8:
            print(f"Error: Found {len(valid_components)} valid-sized animals in card {idx} instead of 8")
            return all_outputs
            
        # Process each valid component (animal)
        for i in valid_components:

            # Get component properties
            x_comp = stats[i, cv2.CC_STAT_LEFT]
            y_comp = stats[i, cv2.CC_STAT_TOP]
            w_comp = stats[i, cv2.CC_STAT_WIDTH]
            h_comp = stats[i, cv2.CC_STAT_HEIGHT]

            # Create component mask
            component_mask = (labels == i).astype(np.uint8) * 255

            # Extract the component
            component_roi = card_roi[y_comp:y_comp+h_comp, x_comp:x_comp+w_comp]
            component_mask = component_mask[y_comp:y_comp+h_comp, x_comp:x_comp+w_comp]

            # Create RGBA image with transparency
            b, g, r = cv2.split(component_roi)
            rgba = cv2.merge([b, g, r, component_mask])

            all_outputs[f"animal_{idx}_{i}"] = rgba

    return all_outputs

def _extract_animal_images_auto_manual(image: np.ndarray, min_area=500):
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

if 1==1:
    _extract_animal_images_auto = _extract_animal_images_auto_advanced
else:
    _extract_animal_images_auto = _extract_animal_images_auto_manual

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

def process_video(input_path: str, output_path: str):
    # 1. 初始化视频读取对象
    cap = cv2.VideoCapture(input_path)

    # 检查视频是否成功打开
    if not cap.isOpened():
        print(f"错误: 无法打开视频文件 {input_path}")
        return

    # 2. 获取原始视频的属性
    # 使用 cap.get() 方法获取帧率 (fps), 帧宽度和高度
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # 3. 定义输出视频的尺寸和编码器
    new_width = 640
    new_height = 480
    output_size = (new_width, new_height)
    
    # 定义视频编码器，'mp4v' 适用于 .mp4 格式
    # 其他选项如 'XVID' 适用于 .avi
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # 4. 初始化视频写入对象
    writer = cv2.VideoWriter(output_path, fourcc, fps, output_size)

    # 5. 循环处理每一帧
    frame_count = 0
    while True:
        # 读取一帧
        ret, frame = cap.read()

        # 如果 ret 为 False，表示视频结束或读取错误
        if not ret:
            break

        images = _extract_animal_images_auto_advanced(frame)

        # if (frame_count % 30 == 0) and (frame_count < 3000):
        #     for k, v in images.items():
        #         image_path = str(Path(output_path).with_suffix("")/f"{frame_count:05d}-{k}.png")
        #         cv2.imwrite(image_path, v)
        #         print(f"Saved debug bounding boxes image to: {image_path}")
        #         print(f"Processing frame {frame_count}")

        # 8. 将处理后的帧写入输出文件
        image = images.get("debug_closed", frame)
        # Ensure the image is resized to the output_size before writing
        image_resized = cv2.resize(image, output_size)
        writer.write(cv2.cvtColor(image_resized, cv2.COLOR_GRAY2RGB))
        
        frame_count += 1

    # 9. 释放资源
    cap.release()
    writer.release()
    
    print(f"处理完成。总共处理了 {frame_count} 帧。")
    print(f"视频已保存至: {output_path}")

if __name__ == "__main__":
    if (Path(__file__).with_name("input.mp4")).exists():
        process_video(str(Path(__file__).with_name("input.mp4")), str(Path(__file__).with_name("output.mp4")))
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
