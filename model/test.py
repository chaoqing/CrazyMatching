

import tensorflow as tf
import numpy as np
import cv2
import os
from pathlib import Path
import argparse

# Constants from training
IMG_HEIGHT = 224
IMG_WIDTH = 224

def draw_predicted_boxes(image, prediction):
    """
    Draws the predicted bounding boxes on a copy of the original image.

    Args:
        image (np.array): The original input image (in BGR format).
        prediction (np.array): The model's output prediction array of shape (10,).
    """
    output_image = image.copy()
    img_h, img_w, _ = output_image.shape

    # Extract the two bounding box details from the prediction
    box1_data = prediction[0:5]
    box2_data = prediction[5:10]

    for i, box_data in enumerate([box1_data, box2_data]):
        # Denormalize the coordinates from the model's output
        center_x = box_data[0] * img_w
        center_y = box_data[1] * img_h
        w = box_data[2] * img_w
        h = box_data[3] * img_h
        rotation_rad = box_data[4]
        rotation_deg = np.degrees(rotation_rad)

        # Get the 4 corners of the rotated rectangle
        # cv2.boxPoints expects ((center_x, center_y), (width, height), angle)
        box_points = cv2.boxPoints(((center_x, center_y), (w, h), rotation_deg))
        box_points = np.intp(box_points)

        # Draw the rectangle on the image
        # Use green for the first box and red for the second for clarity
        color = (0, 255, 0) if i == 0 else (0, 0, 255)
        cv2.drawContours(output_image, [box_points], 0, color, thickness=2)

    return output_image

def run_inference(model_path, image_path, output_dir):
    """
    Loads a saved model, runs inference on an image, and saves the result.
    """
    # 1. Load the saved model
    print(f"Loading model from: {model_path}")
    try:
        model = tf.saved_model.load(str(model_path))
    except OSError:
        print(f"Error: Model not found at '{model_path}'.")
        print("Please ensure you have trained the model by running `python model/train.py` first.")
        return

    # 2. Load and preprocess the input image
    print(f"Loading image: {image_path}")
    if not image_path.exists():
        print(f"Error: Image file not found at '{image_path}'")
        return
        
    original_image = cv2.imread(str(image_path))
    if original_image is None:
        print(f"Error: Could not read the image file '{image_path}'")
        return

    # Preprocess for the model: resize, convert color, normalize
    img_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (IMG_WIDTH, IMG_HEIGHT))
    img_normalized = img_resized.astype(np.float32) / 255.0
    
    # Add a batch dimension
    input_tensor = np.expand_dims(img_normalized, axis=0)

    # 3. Run prediction
    print("Running inference...")
    # The loaded model is a callable function
    prediction = model(input_tensor)
    
    # The output is a tensor; convert it to a NumPy array and remove the batch dim
    prediction_array = prediction.numpy().squeeze()
    print(f"Model prediction (normalized): {prediction_array}")

    # 4. Draw bounding boxes on the original image
    output_image = draw_predicted_boxes(original_image, prediction_array)

    # 5. Save the result
    os.makedirs(output_dir, exist_ok=True)
    output_filename = f"{image_path.stem}_prediction.png"
    output_path = output_dir / output_filename
    cv2.imwrite(str(output_path), output_image)
    print(f"\nInference complete. Result saved to: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run inference on an image using the trained Crazy Matching model.")
    parser.add_argument(
        '--image',
        type=str,
        required=True,
        help="Path to the input image file."
    )
    args = parser.parse_args()

    # Define paths
    base_dir = Path(__file__).parent
    model_path = base_dir / 'saved_model' / 'my_custom_object_detection_model'
    image_path = Path(args.image)
    output_dir = base_dir / 'test_output'

    run_inference(model_path, image_path, output_dir)

