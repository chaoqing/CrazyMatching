import cv2
import numpy as np
import os
from pathlib import Path
import random
import xml.etree.ElementTree as ET

# Define the classes based on the animal_XXX.png files
CLASSES = [
    'animal_1010', 'animal_1016', 'animal_1089', 'animal_1236', 'animal_1400',
    'animal_1478', 'animal_1492', 'animal_543', 'animal_639', 'animal_689',
    'animal_806', 'animal_859', 'animal_913', 'animal_951', 'animal_986'
]

def overlay_image(background, overlay, x, y):
    """
    Overlays a transparent RGBA image onto a BGR background image at a specific location.
    """
    # Ensure overlay is 4 channels
    if overlay.shape[2] == 3:
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
        overlay_x2 -= (x2 - b_w)
        x2 = b_w
    if y2 > b_h:
        overlay_y2 -= (y2 - b_h)
        y2 = b_h

    # Get the region of interest on the background
    roi = background[y1:y2, x1:x2]
    
    # Crop the overlay image to match the clipped ROI
    overlay_cropped = overlay[overlay_y1:overlay_y2, overlay_x1:overlay_x2]

    # Ensure the ROI and overlay have the same dimensions
    if roi.shape[0] != overlay_cropped.shape[0] or roi.shape[1] != overlay_cropped.shape[1]:
        return background

    # Split overlay into color and alpha channels
    b, g, r, a = cv2.split(overlay_cropped)
    overlay_rgb = cv2.merge((b, g, r))
    
    # Normalize alpha mask to be between 0 and 1
    alpha = a / 255.0
    alpha = np.expand_dims(alpha, axis=2) # for broadcasting

    # Perform alpha blending
    blended_roi = (alpha * overlay_rgb + (1 - alpha) * roi).astype(np.uint8)

    background[y1:y2, x1:x2] = blended_roi
    return background

def create_pascal_voc_xml(image_path, annotations, img_width, img_height, img_depth=3):
    """
    Creates a Pascal VOC XML annotation file.
    Args:
        image_path (Path): Path to the image file.
        annotations (list): List of dictionaries, each containing 'class', 'xmin', 'ymin', 'xmax', 'ymax'.
        img_width (int): Width of the image.
        img_height (int): Height of the image.
        img_depth (int): Depth of the image (e.g., 3 for RGB).
    Returns:
        ET.Element: The root element of the XML tree.
    """
    annotation = ET.Element('annotation')
    
    folder = ET.SubElement(annotation, 'folder')
    folder.text = image_path.parent.name # e.g., 'images'

    filename = ET.SubElement(annotation, 'filename')
    filename.text = image_path.name

    path = ET.SubElement(annotation, 'path')
    path.text = str(image_path.resolve())

    source = ET.SubElement(annotation, 'source')
    database = ET.SubElement(source, 'database')
    database.text = 'CrazyMatching Dataset'

    size = ET.SubElement(annotation, 'size')
    width = ET.SubElement(size, 'width')
    width.text = str(img_width)
    height = ET.SubElement(size, 'height')
    height.text = str(img_height)
    depth = ET.SubElement(size, 'depth')
    depth.text = str(img_depth)

    segmented = ET.SubElement(annotation, 'segmented')
    segmented.text = '0'

    for ann in annotations:
        obj = ET.SubElement(annotation, 'object')
        name = ET.SubElement(obj, 'name')
        name.text = ann['class']
        pose = ET.SubElement(obj, 'pose')
        pose.text = 'Unspecified'
        truncated = ET.SubElement(obj, 'truncated')
        truncated.text = '0'
        difficult = ET.SubElement(obj, 'difficult')
        difficult.text = '0'
        bndbox = ET.SubElement(obj, 'bndbox')
        xmin = ET.SubElement(bndbox, 'xmin')
        xmin.text = str(ann['xmin'])
        ymin = ET.SubElement(bndbox, 'ymin')
        ymin.text = str(ann['ymin'])
        xmax = ET.SubElement(bndbox, 'xmax')
        xmax.text = str(ann['xmax'])
        ymax = ET.SubElement(bndbox, 'ymax')
        ymax.text = str(ann['ymax'])
    
    return ET.ElementTree(annotation)

def generate_tfod_data(num_samples=100, output_base_dir='.', original_assets_dir='/teamspace/studios/this_studio/Work/CrazyMatching/model/data'):
    """
    Generates simulated training data for TFOD in Pascal VOC format.
    """
    output_base_dir = Path(output_base_dir)
    images_dir = output_base_dir / 'images'
    annotations_dir = output_base_dir / 'annotations'
    debug_output_dir = output_base_dir / 'debug_output'

    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(annotations_dir, exist_ok=True)
    os.makedirs(debug_output_dir, exist_ok=True)

    original_assets_dir = Path(original_assets_dir)
    animal_paths = {p.stem: p for p in original_assets_dir.glob('animal_*.png')}
    background_path = original_assets_dir / 'background.png'
    
    if not animal_paths:
        print("Error: No animal images found in original assets directory.")
        return
        
    if not background_path.exists():
        print(f"Error: Background image not found at {background_path}")
        return

    background_img = cv2.imread(str(background_path))
    if background_img is None:
        print(f"Error: Could not read background image from {background_path}")
        return
        
    bg_h, bg_w, _ = background_img.shape

    print(f"Starting data generation for {num_samples} samples...")

    for i in range(num_samples):
        sample_image = background_img.copy()
        annotations = []
        placed_objects = [] # To keep track of placed objects for collision detection

        num_animals_to_place = random.randint(5, 10) # Place 5 to 10 animals per image

        attempts = 0
        while len(placed_objects) < num_animals_to_place and attempts < 200:
            attempts += 1
            
            # Randomly select an animal class and its image
            animal_name = random.choice(CLASSES)
            animal_path = animal_paths[animal_name]
            animal_img = cv2.imread(str(animal_path), cv2.IMREAD_UNCHANGED)
            if animal_img is None: continue

            # Random scale for the animal
            scale = random.uniform(0.6, 1.0) # Smaller scale for more items
            
            # Calculate target size based on scale
            original_h, original_w = animal_img.shape[:2]
            target_w = int(original_w * scale)
            target_h = int(original_h * scale)

            if target_w == 0 or target_h == 0: continue # Avoid zero-sized images

            resized_animal = cv2.resize(animal_img, (target_w, target_h))

            # Random position for the animal
            # Ensure the entire animal fits within the background
            max_x = bg_w - target_w
            max_y = bg_h - target_h
            if max_x <= 0 or max_y <= 0: continue # Animal is too large for background

            x = random.randint(0, max_x)
            y = random.randint(0, max_y)

            # Simple collision detection (bounding box overlap)
            is_overlapping = False
            for obj_x, obj_y, obj_w, obj_h in placed_objects:
                if not (x + target_w < obj_x or \
                        x > obj_x + obj_w or \
                        y + target_h < obj_y or \
                        y > obj_y + obj_h):
                    is_overlapping = True
                    break
            
            if not is_overlapping:
                overlay_image(sample_image, resized_animal, x, y)
                placed_objects.append((x, y, target_w, target_h))
                
                annotations.append({
                    'class': animal_name,
                    'xmin': x,
                    'ymin': y,
                    'xmax': x + target_w,
                    'ymax': y + target_h
                })

        if not annotations:
            print(f"Warning: No objects placed for sample {i}. Skipping.")
            continue

        # Save image
        image_filename = f"sample_{i:04d}.jpg"
        image_path = images_dir / image_filename
        cv2.imwrite(str(image_path), sample_image)

        # Save XML annotation
        xml_filename = f"sample_{i:04d}.xml"
        xml_path = annotations_dir / xml_filename
        tree = create_pascal_voc_xml(image_path, annotations, bg_w, bg_h)
        tree.write(str(xml_path))
        
        if (i + 1) % 100 == 0:
            print(f"  Generated {i + 1}/{num_samples} samples...")

    print(f"\nData generation complete. Generated {num_samples} samples.")
    print(f"Images saved in: '{images_dir}'")
    print(f"Annotations saved in: '{annotations_dir}'")

def debug_tfod_data(image_path, annotation_path, output_path):
    """
    Reads an image and its Pascal VOC XML annotation, draws bounding boxes,
    and saves the debug image.
    """
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return

    tree = ET.parse(str(annotation_path))
    root = tree.getroot()

    debug_image = image.copy()

    for obj in root.findall('object'):
        name = obj.find('name').text
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)

        cv2.rectangle(debug_image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(debug_image, name, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(str(output_path), debug_image)
    print(f"Saved debug image with bounding boxes to: {output_path}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Generate simulated training data for TFOD.")
    parser.add_argument('--num_samples', type=int, default=100,
                        help='Number of samples to generate.')
    parser.add_argument('--output_base_dir', type=str, default='.',
                        help='Base directory for output images, annotations, and debug_output.')
    parser.add_argument('--original_assets_dir', type=str,
                        default='/teamspace/studios/this_studio/Work/CrazyMatching/model/data',
                        help='Directory containing original animal PNGs and background.png.')
    args = parser.parse_args()

    output_dir = Path(args.output_base_dir)
    print(f"Output directory: {output_dir}")

    # --- Part 1: Generate simulated training data for TFOD ---
    print("--- Running Step 1: Generating TFOD Training Data ---")
    generate_tfod_data(num_samples=args.num_samples,
                       output_base_dir=str(output_dir),
                       original_assets_dir=args.original_assets_dir)
    print("\n--- Finished Step 1 ---")

    # --- Part 2: Debug and verify a training sample ---
    print("\n--- Running Step 2: Debugging a Training Sample ---")
    images_dir = output_dir / 'images'
    annotations_dir = output_dir / 'annotations'
    debug_output_dir = output_dir / 'debug_output'

    image_files = list(images_dir.glob('*.jpg'))

    if not image_files:
        print("No training images found. Please run Step 1 to generate data first.")
    else:
        # Pick a random image to debug
        random_image_path = random.choice(image_files)
        annotation_path = annotations_dir / (random_image_path.stem + '.xml')
        debug_output_path = debug_output_dir / random_image_path.name

        if annotation_path.exists():
            debug_tfod_data(random_image_path, annotation_path, debug_output_path)
        else:
            print(f"Error: Could not find corresponding annotation for {random_image_path.name}")
    print("\n--- Finished Step 2 ---")
