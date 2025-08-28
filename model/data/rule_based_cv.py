import cv2
import numpy as np
import os
from pathlib import Path
import random
import json
from tqdm import tqdm

def compute_similarity_ORB(image1: np.ndarray, image2: np.ndarray) -> float:
    """
    Compute similarity between two RGBA images that is invariant to size and rotation.
    
    Args:
        image1: First RGBA image as numpy array
        image2: Second RGBA image as numpy array
        
    Returns:
        float: Similarity score between 0 and 1, where 1 means identical
    """
    # Convert RGBA to grayscale for feature detection
    if image1.shape[-1] == 4:  # RGBA
        image1_gray = cv2.cvtColor(image1, cv2.COLOR_RGBA2GRAY)
        image2_gray = cv2.cvtColor(image2, cv2.COLOR_RGBA2GRAY)
    else:  # Assume RGB
        image1_gray = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
        image2_gray = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)
    
    # Initialize ORB detector
    orb = cv2.ORB_create(nfeatures=1000)
    
    # Detect keypoints and compute descriptors
    keypoints1, descriptors1 = orb.detectAndCompute(image1_gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(image2_gray, None)
    
    # If no keypoints found, images are likely blank or too similar
    if descriptors1 is None or descriptors2 is None:
        return 0.0
    
    # Create BF (Brute Force) matcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    # Match descriptors
    matches = bf.match(descriptors1, descriptors2)
    
    # Sort matches by distance (lower distance = better match)
    matches = sorted(matches, key=lambda x: x.distance)
    
    # Calculate similarity score
    # Consider both the number of good matches and their quality
    max_matches = min(len(keypoints1), len(keypoints2))
    if max_matches == 0:
        return 0.0
        
    # Get the distances of all matches
    distances = np.array([m.distance for m in matches])
    
    # Consider only good matches (distance < 64 is typically good for ORB)
    good_matches = distances < 64
    num_good_matches = np.sum(good_matches)
    
    # Calculate similarity score based on number of good matches and their quality
    if num_good_matches == 0:
        return 0.0
        
    # Normalize by the maximum possible matches
    similarity = num_good_matches / max_matches
    
    # Adjust score based on average distance of good matches
    if len(good_matches) > 0:
        avg_distance = np.mean(distances[good_matches])
        distance_factor = 1 - (avg_distance / 64)  # Normalize to [0,1]
        similarity = similarity * distance_factor
    
    return min(1.0, max(0.0, similarity))

def compute_similarity_simple(image1: np.ndarray, image2: np.ndarray) -> float:
    """
    Compute similarity between two RGBA images using Hu Moments.
    This method is invariant to translation, rotation, and scale.
    
    Args:
        image1: First RGBA image as numpy array
        image2: Second RGBA image as numpy array
        
    Returns:
        float: Similarity score between 0 and 1, where 1 means identical
    """
    # Convert RGBA to grayscale
    if image1.shape[-1] == 4:  # RGBA
        image1_gray = cv2.cvtColor(image1, cv2.COLOR_RGBA2GRAY)
        image2_gray = cv2.cvtColor(image2, cv2.COLOR_RGBA2GRAY)
    else:  # Assume RGB
        image1_gray = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
        image2_gray = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)
    
    # Calculate Hu Moments for both images
    moments1 = cv2.moments(image1_gray)
    moments2 = cv2.moments(image2_gray)
    
    # Handle empty images or failed moment calculation
    if moments1['m00'] == 0 or moments2['m00'] == 0:
        return 0.0
    
    # Calculate Hu Moments
    hu_moments1 = cv2.HuMoments(moments1)
    hu_moments2 = cv2.HuMoments(moments2)
    
    # Convert to log scale to handle small values better
    for i in range(7):
        if hu_moments1[i] != 0:
            hu_moments1[i] = -np.sign(hu_moments1[i]) * np.log10(abs(hu_moments1[i]))
        if hu_moments2[i] != 0:
            hu_moments2[i] = -np.sign(hu_moments2[i]) * np.log10(abs(hu_moments2[i]))
    
    # Calculate similarity using normalized L2 distance between Hu moments
    distance = np.linalg.norm(hu_moments1 - hu_moments2)
    
    # Convert distance to similarity score (0 to 1)
    # Using an exponential decay function: similarity = exp(-distance/scale)
    # Scale factor of 2.0 chosen empirically - adjust if needed
    similarity = np.exp(-distance/2.0)
    
    return float(similarity)

if 1 == 0:
    compute_similarity = compute_similarity_ORB
else:
    compute_similarity = compute_similarity_simple

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

    all_outputs["input"] = image

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
    min_card_area = image.shape[0] * image.shape[1] * 0.05  # Cards should be at least 10% of image
    
    # Create debug image for all contours
    debug_all_contours = image.copy()
    cv2.drawContours(debug_all_contours, contours, -1, (0, 0, 255), 2)  # Draw all contours in red
    all_outputs["debug_all_contours"] = debug_all_contours
    
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
            
        # Add visualization of the approximated contour
        debug_approx = image.copy()
        cv2.drawContours(debug_approx, [approx], -1, (255, 0, 0), 2)  # Draw approximated contour in blue
        all_outputs[f"debug_approx_contour_{len(card_contours)}"] = debug_approx
    
    # We need at least 2 card contours
    if len(card_contours) < 2:
        print(f"Error: Found {len(card_contours)} cards instead of at least 2")
        return all_outputs
        
    # If we have more than 2 contours, find the pair with most similar areas
    # Get areas of all contours
    areas = [cv2.contourArea(cnt) for cnt in card_contours]

    # Find the pair of contours with the most similar areas
    min_diff_ratio = float('inf')
    best_pair = (0, 1)
    
    for i in range(len(card_contours)):
        for j in range(i + 1, len(card_contours)):
            area_diff_ratio = abs(areas[i] - areas[j]) / max(areas[i], areas[j])
            if area_diff_ratio < min_diff_ratio:
                min_diff_ratio = area_diff_ratio
                best_pair = (i, j)
    
    if min_diff_ratio > 0.2:  # More than 20% difference
        print(f"Error: Card areas differ too much: {area_diff_ratio:.2%}")
        return all_outputs
    
    # Keep only the two most similar contours
    card_contours = [card_contours[best_pair[0]], card_contours[best_pair[1]]]

    # Debug image for card detection
    debug_cards = image.copy()
    cv2.drawContours(debug_cards, card_contours, -1, (0, 255, 0), 2)
    all_outputs["debug_card_detection"] = debug_cards

    all_animals = ({}, {})

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

        # Save the extracted card region for debugging
        all_outputs[f"debug_card_{idx}_region"] = card_roi

        # Convert to grayscale for adaptive thresholding
        gray_roi = cv2.cvtColor(card_roi, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding to handle varying lighting conditions
        block_size = 25  # Must be odd
        C = 10  # Constant subtracted from mean
        white_mask = cv2.adaptiveThreshold(
            gray_roi,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size,
            C
        )
        
        # Save white mask for debugging
        all_outputs[f"debug_card_{idx}_white_mask"] = white_mask
        
        # Invert to get the animals
        animal_mask = cv2.bitwise_not(white_mask)
        
        # Close small gaps in the animal mask using morphological operations
        small_kernel = np.ones((3,3), np.uint8)  # Smaller kernel for fine details
        # First close small gaps
        animal_mask = cv2.morphologyEx(animal_mask, cv2.MORPH_CLOSE, small_kernel, iterations=2)
        # Save intermediate result after closing small gaps
        all_outputs[f"debug_card_{idx}_closed_gaps"] = animal_mask
        
        # Find initial components before cleanup
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            animal_mask, connectivity=8
        )
        
        # Create debug visualization of initial components
        debug_initial = np.zeros((animal_mask.shape[0], animal_mask.shape[1], 3), dtype=np.uint8)
        for i in range(1, num_labels):
            # Random color for each component
            color = np.random.randint(0, 255, size=3).tolist()
            debug_initial[labels == i] = color
        all_outputs[f"debug_card_{idx}_initial_components"] = debug_initial
        
        # Process each component to fill holes and calculate solid area
        filled_components = []
        card_area = animal_mask.shape[0] * animal_mask.shape[1]
        min_component_area = 0.001 * card_area  # 5% of card area threshold
        max_component_area = 0.5 * card_area  # 50% of card area threshold
        
        for i in range(1, num_labels):
            # Create mask for this component
            component_mask = (labels == i).astype(np.uint8) * 255
            
            # Find contours to fill holes
            contours, _ = cv2.findContours(
                component_mask, 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            # Create a mask from convex hull
            hull_mask = np.zeros_like(component_mask)
            for contour in contours:
                # Find convex hull points
                hull = cv2.convexHull(contour)
                # Draw filled convex hull
                cv2.fillPoly(hull_mask, [hull], 255)
            
            filled_hull_mask = cv2.bitwise_and(hull_mask, animal_mask)
            large_kernel = np.ones((10,10), np.uint8)
            component_mask = cv2.morphologyEx(filled_hull_mask, cv2.MORPH_CLOSE, large_kernel, iterations=2)
            component_mask = cv2.bitwise_and(hull_mask, component_mask)
            contours, _ = cv2.findContours(
                component_mask, 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )

            filled_mask = np.zeros_like(component_mask)
            cv2.drawContours(filled_mask, contours, -1, (255), cv2.FILLED)
            
            # Calculate solid area after filling
            solid_area = np.count_nonzero(hull_mask)

            # Only keep components smaller than threshold
            if solid_area < max_component_area and solid_area > min_component_area:
                all_outputs[f"debug_card_{idx}.{i}_hull_mask"] = hull_mask
                filled_components.append((filled_mask, solid_area))
        
        # Sort components by area and take top 8
        filled_components.sort(key=lambda x: x[1], reverse=True)
        filled_components = filled_components[:8]
        
        # Create new animal mask from filled components
        animal_mask = np.zeros_like(animal_mask)
        debug_filled = np.zeros((animal_mask.shape[0], animal_mask.shape[1], 3), dtype=np.uint8)
        
        for i, (filled_mask, area) in enumerate(filled_components):
            animal_mask = cv2.bitwise_or(animal_mask, filled_mask)
            # Add colored visualization of filled components
            color = np.random.randint(0, 255, size=3).tolist()
            debug_filled[filled_mask > 0] = color
            
        # Save debug visualization of filled components
        all_outputs[f"debug_card_{idx}_filled_components"] = debug_filled
        
        # We already have the filled components from earlier processing
        # Use the animal_mask we created from filled components
        all_outputs[f"debug_card_{idx}_animal_mask"] = animal_mask
        
        # Store the card region and animal mask for final combination
        all_animals[idx-1]["card_region"] = (x, y, w, h)
        all_animals[idx-1]["animal_mask"] = cv2.cvtColor(debug_filled, cv2.COLOR_BGR2GRAY)
        
        # Find components on the filled mask - these should already be our 8 animals
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            animal_mask, connectivity=8
        )
        
        # Create a visualization of the animal mask overlaid on the card
        animal_overlay = card_roi.copy()
        animal_overlay[animal_mask > 0] = [0, 255, 0]  # Highlight animals in green
        all_outputs[f"debug_card_{idx}_animal_overlay"] = animal_overlay
        
        # Find animal components on the filled mask
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            animal_mask, connectivity=8
        )
        
        # Create a colorful visualization of the connected components
        label_hue = np.uint8(179 * labels / np.max(labels))
        blank_ch = 255 * np.ones_like(label_hue)
        labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
        labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
        labeled_img[label_hue == 0] = 0  # Set background to black
        all_outputs[f"debug_card_{idx}_components"] = labeled_img

        if num_labels not in range(7, 11):
            print(f"Skip card {idx} because {num_labels-1} animals detected")
            continue
        
        valid_components = list(range(1, num_labels))                
            
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

            all_animals[idx-1].setdefault("final_animal_mask", []).append({"animal_region": [x_comp+x, y_comp+y, w_comp, h_comp], "animal_img": rgba})
    
    if all(all_animals[card_idx-1].get("final_animal_mask", None) is None for card_idx in [1, 2]):
        print(f"No valid animals found in cards 1 or 2")
        return all_outputs

    # Create final combined animal mask on the original image
    final_mask = np.zeros_like(gray)
    # Create RGBA image with transparency - same size as input image but with alpha channel
    final_image = cv2.merge([*cv2.split(image), np.zeros_like(gray)])# Start with fully transparent image

    # Combine masks from both cards
    for card_idx in [1, 2]:
        if all_animals[card_idx-1]:
            x, y, w, h = all_animals[card_idx-1]["card_region"]
            card_mask = all_animals[card_idx-1]["animal_mask"]
            # Place the card's animal mask in the correct position on the final mask
            final_mask[y:y+h, x:x+w] = cv2.bitwise_or(
                final_mask[y:y+h, x:x+w],
                card_mask
            )

            # Add each detected animal to the final image
            for patch in all_animals[card_idx-1].get("final_animal_mask", []):
                x, y, w, h = patch["animal_region"]
                animal_img = patch["animal_img"]
                final_image[y:y+h, x:x+w] = animal_img

    # Add the final results to the outputs
    all_outputs["final_animal_mask"] = final_mask
    all_outputs["final_animal_image"] = final_image

    if not all(len(all_animals[card_idx-1].get("final_animal_mask", []))==8 for card_idx in [1, 2]):
        print(f"Each card must have exactly 8 animals detected.")
        return all_outputs

    # for each final_animal_mask in the two cards, create a 8x8 matrix to quantily the similarity of the pair of patch.
    max_smilarity = -float('inf')
    best_pair = (0, 0)
    
    for i, animal_i in enumerate(all_animals[0]["final_animal_mask"]):
        for j, animal_j in enumerate(all_animals[1]["final_animal_mask"]):
            if i>j: continue

            similarity = compute_similarity(animal_i["animal_img"], animal_j["animal_img"])
            if max_smilarity < similarity:
                max_smilarity = similarity
                best_pair = (i, j)
        
    final_animal_image_with_boxes = image.copy()
    # overlay the bbox with red on final_image
    x1, y1, w1, h1 = all_animals[0]["final_animal_mask"][best_pair[0]]["animal_region"]
    x2, y2, w2, h2 = all_animals[1]["final_animal_mask"][best_pair[1]]["animal_region"]
    cv2.rectangle(final_animal_image_with_boxes, (x1, y1), (x1 + w1, y1 + h1), (0, 0, 255, 255), 2)
    cv2.rectangle(final_animal_image_with_boxes, (x2, y2), (x2 + w2, y2 + h2), (0, 0, 255, 255), 2)
    all_outputs["final_animal_image_with_boxes"] = final_animal_image_with_boxes

    print(f"final detection result {best_pair} - {max_smilarity}: ({x1}, {y1}, {w1}, {h1}, 0, {x2}, {y2}, {w2}, {h2}, 0)")

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

    for i, (k, v) in enumerate(_extract_animal_images_auto(image, min_area=min_area).items()):
        image_path = os.path.join(output_dir, f"{i:02d}.{k}.png")
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
    new_width = 480
    new_height = 640
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

        if (frame_count % round(fps) == 0) and (frame_count < fps * 60):  # 每秒保存一帧，最多保存前10秒的调试图像
            for k, v in images.items():
                if k not in ("input", "debug_all_contours", "final_animal_mask"): continue
                image_path = str(Path(output_path).with_suffix("")/f"{frame_count:05d}-{k}.png")
                cv2.imwrite(image_path, v)
                print(f"Saved debug bounding boxes image to: {image_path}")
                print(f"Processing frame {frame_count}")

        if "final_animal_image" in images:
            image_path = str(Path(output_path).with_suffix("")/f"{frame_count:05d}-final_animal_image.png")
            cv2.imwrite(image_path, images["final_animal_image"])

        # 8. 将处理后的帧写入输出文件
        image = next((images[i] for i in ("final_animal_image_with_boxes", "final_animal_image", "debug_all_contours", "input") if i in images), frame)

        # Ensure the image is resized to the output_size before writing
        image_resized = cv2.resize(image, output_size)
        if image_resized.ndim == 2:  # Check if the image is grayscale
            image_resized = cv2.cvtColor(image_resized, cv2.COLOR_GRAY2BGR)
        if image_resized.ndim == 4:  # Check if the image is RGBA
            image_resized = cv2.cvtColor(image_resized, cv2.COLOR_RGBA2BGR)
        writer.write(image_resized)

        frame_count += 1

    # 9. 释放资源
    cap.release()
    writer.release()
    
    print(f"处理完成。总共处理了 {frame_count} 帧。")
    print(f"视频已保存至: {output_path}")

if __name__ == "__main__":
    image_file = Path(__file__).with_name("..")/'../public/example.jpg'
    output_directory_extract = Path(__file__).with_name('extracted_animals')
    extract_animal_images_auto(image_file, output_directory_extract, min_area=1000)

    if (Path(__file__).with_name("input.mp4")).exists():
        Path(__file__).with_name("output").mkdir(exist_ok=True)
        process_video(str(Path(__file__).with_name("input.mp4")), str(Path(__file__).with_name("output.mp4")))

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
