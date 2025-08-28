import cv2
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
import argparse

def compute_similarity(image1: np.ndarray, image2: np.ndarray) -> float:
    """
    Compute similarity between two RGBA images using Hu Moments.
    This method is invariant to translation, rotation, and scale.
    
    Args:
        image1: First RGBA image as numpy array
        image2: Second RGBA image as numpy array
        
    Returns:
        float: Similarity score between 0 and 1, where 1 means identical
    """
    # Convert to grayscale
    if image1.shape[-1] == 4:  # RGBA
        image1_gray = cv2.cvtColor(image1, cv2.COLOR_RGBA2GRAY)
        image2_gray = cv2.cvtColor(image2, cv2.COLOR_RGBA2GRAY)
    else:  # Assume RGB/BGR
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
    similarity = np.exp(-distance / 2.0)
    
    return float(similarity)

def find_similar_patches(image: np.ndarray):
    """
    Find two cards, extract patches from each, and find the most similar pair.
    
    Args:
        image: BGR format image containing two cards
        
    Returns:
        Tuple: (card1_bbox, card2_bbox, patches1, patches2, best_pair, similarity)
        or None if detection fails
    """
    if image is None or image.size == 0:
        return None

    # Preprocessing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
    edges = cv2.Canny(bilateral, 50, 150)
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=2)
    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Find contours for cards
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_card_area = image.shape[0] * image.shape[1] * 0.05
    card_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_card_area:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            card_contours.append(cnt)

    if len(card_contours) < 2:
        return None

    # Find pair with most similar areas, which should be the two cards
    areas = [cv2.contourArea(cnt) for cnt in card_contours]
    min_diff_ratio = float('inf')
    best_pair_idx = (0, 1)
    for i in range(len(card_contours)):
        for j in range(i + 1, len(card_contours)):
            diff_ratio = abs(areas[i] - areas[j]) / max(areas[i], areas[j])
            if diff_ratio < min_diff_ratio:
                min_diff_ratio = diff_ratio
                best_pair_idx = (i, j)

    if min_diff_ratio > 0.2:
        return None

    card_contours = [card_contours[best_pair_idx[0]], card_contours[best_pair_idx[1]]]
    bboxes = [cv2.boundingRect(c) for c in card_contours]
    card1_bbox, card2_bbox = bboxes 

    # Process each card
    patches = [[], []]
    for card_idx, card_cnt in enumerate(card_contours):
        # Extract card region
        x, y, w, h = bboxes[card_idx]
        card_mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(card_mask, [card_cnt], -1, 255, -1)
        card_region = cv2.bitwise_and(image, image, mask=card_mask)
        card_roi = card_region[y:y + h, x:x + w]
        gray_roi = cv2.cvtColor(card_roi, cv2.COLOR_BGR2GRAY)

        # Adaptive thresholding
        white_mask = cv2.adaptiveThreshold(
            gray_roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 10
        )
        animal_mask = cv2.bitwise_not(white_mask)

        # Morphological close
        small_kernel = np.ones((3, 3), np.uint8)
        animal_mask = cv2.morphologyEx(animal_mask, cv2.MORPH_CLOSE, small_kernel, iterations=2)

        # Connected components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(animal_mask, connectivity=8)

        filled_components = []
        card_area = w * h
        min_component_area = 0.001 * card_area
        max_component_area = 0.5 * card_area

        for i in range(1, num_labels):
            component_mask = (labels == i).astype(np.uint8) * 255
            contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            hull_mask = np.zeros_like(component_mask)
            for cont in contours:
                hull = cv2.convexHull(cont)
                cv2.fillPoly(hull_mask, [hull], 255)
            filled_hull_mask = cv2.bitwise_and(hull_mask, animal_mask)
            large_kernel = np.ones((10, 10), np.uint8)
            component_mask = cv2.morphologyEx(filled_hull_mask, cv2.MORPH_CLOSE, large_kernel, iterations=2)
            component_mask = cv2.bitwise_and(hull_mask, component_mask)
            contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            filled_mask = np.zeros_like(component_mask)
            cv2.drawContours(filled_mask, contours, -1, 255, cv2.FILLED)
            solid_area = np.count_nonzero(hull_mask)
            if min_component_area < solid_area < max_component_area:
                filled_components.append((filled_mask, solid_area))

        # Sort and rebuild animal_mask from all valid components
        filled_components.sort(key=lambda x: x[1], reverse=True)
        animal_mask = np.zeros_like(animal_mask)
        for filled_mask, _ in filled_components:
            animal_mask = cv2.bitwise_or(animal_mask, filled_mask)

        # Final connected components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(animal_mask, connectivity=8)
        for i in range(1, num_labels):
            x_comp = stats[i, cv2.CC_STAT_LEFT]
            y_comp = stats[i, cv2.CC_STAT_TOP]
            w_comp = stats[i, cv2.CC_STAT_WIDTH]
            h_comp = stats[i, cv2.CC_STAT_HEIGHT]
            component_mask_full = (labels == i).astype(np.uint8) * 255
            component_roi = card_roi[y_comp:y_comp + h_comp, x_comp:x_comp + w_comp]
            component_mask = component_mask_full[y_comp:y_comp + h_comp, x_comp:x_comp + w_comp]
            b, g, r = cv2.split(component_roi)
            rgba = cv2.merge([b, g, r, component_mask])
            abs_bbox = (x + x_comp, y + y_comp, w_comp, h_comp)
            patches[card_idx].append({'bbox': abs_bbox, 'img': rgba, 'mask': component_mask})

    if not patches[0] or not patches[1]:
        return card1_bbox, card2_bbox, patches[0], patches[1], None, 0.0

    # Find best matching pair
    max_similarity = -float('inf')
    best_pair = (0, 0)
    for i, p1 in enumerate(patches[0]):
        for j, p2 in enumerate(patches[1]):
            similarity = compute_similarity(p1['img'], p2['img'])
            if similarity > max_similarity:
                max_similarity = similarity
                best_pair = (i, j)

    return card1_bbox, card2_bbox, patches[0], patches[1], best_pair, max_similarity

def draw_dashed_rect(img, pt1, pt2, color, thickness=1, dash_len=10):
    x1, y1 = pt1
    x2, y2 = pt2
    # Top
    for i in range(x1, x2, dash_len * 2):
        cv2.line(img, (i, y1), (min(i + dash_len, x2), y1), color, thickness)
    # Bottom
    for i in range(x1, x2, dash_len * 2):
        cv2.line(img, (i, y2), (min(i + dash_len, x2), y2), color, thickness)
    # Left
    for i in range(y1, y2, dash_len * 2):
        cv2.line(img, (x1, i), (x1, min(i + dash_len, y2)), color, thickness)
    # Right
    for i in range(y1, y2, dash_len * 2):
        cv2.line(img, (x2, i), (x2, min(i + dash_len, y2)), color, thickness)

def process_image(image: np.ndarray):
    """
    Process the image to find similar patches and visualize.
    
    Args:
        image: Input BGR image
        
    Returns:
        vis_image (np.ndarray), similarity (float)
    """
    results = find_similar_patches(image)
    vis = image.copy()
    sim = 0.0

    if results is None:
        return vis, sim

    card1_bbox, card2_bbox, patches1, patches2, best_pair, sim = results

    # Draw dashed bboxes for cards (blue)
    draw_dashed_rect(vis, (card1_bbox[0], card1_bbox[1]),
                     (card1_bbox[0] + card1_bbox[2], card1_bbox[1] + card1_bbox[3]), (255, 0, 0), 2)
    draw_dashed_rect(vis, (card2_bbox[0], card2_bbox[1]),
                     (card2_bbox[0] + card2_bbox[2], card2_bbox[1] + card2_bbox[3]), (255, 0, 0), 2)

    # Draw solid bboxes for animals (green)
    for p in patches1 + patches2:
        bbox = p['bbox']
        cv2.rectangle(vis, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 0), 1)

    # Overlay contours for matched pair (red)
    if best_pair is not None:
        p1 = patches1[best_pair[0]]
        p2 = patches2[best_pair[1]]
        contours1, _ = cv2.findContours(p1['mask'], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        x1, y1, _, _ = p1['bbox']
        cv2.drawContours(vis, contours1, -1, (0, 0, 255), 2, offset=(x1, y1))
        contours2, _ = cv2.findContours(p2['mask'], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        x2, y2, _, _ = p2['bbox']
        cv2.drawContours(vis, contours2, -1, (0, 0, 255), 2, offset=(x2, y2))

    return vis, sim

def process_image_path(input_path: str, output_dir: Path):
    image = cv2.imread(input_path)
    if image is None:
        return
    vis, sim = process_image(image)
    stem = Path(input_path).stem
    sim_perc = int(sim * 100)
    output_path = output_dir / f"{stem}_{sim_perc}.png"
    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(str(output_path), vis)

def process_video_path(input_path: str, output_path: str):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for _ in tqdm(range(65535), desc="Processing video frames"):
        ret, frame = cap.read()
        if not ret:
            break
        vis, _ = process_image(frame)
        writer.write(vis)

    cap.release()
    writer.release()

def main():
    parser = argparse.ArgumentParser(description="Process image or video to find similar patches on cards.")
    parser.add_argument("input", type=str, help="Input image or video path")
    parser.add_argument("output", type=str, help="Output directory (for image) or file path (for video)")
    args = parser.parse_args()

    input_path = Path(args.input)
    output = Path(args.output)

    if input_path.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp'):
        output_dir = output if output.is_dir() else output.parent
        process_image_path(str(input_path), output_dir)
    elif input_path.suffix.lower() in ('.mp4', '.avi'):
        if output.is_dir():
            output = output / "output.mp4"
        process_video_path(str(input_path), str(output))
    else:
        print("Unsupported input file type.")

if __name__ == "__main__":
    main()
