

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

def run_inference(model_path: Path, image_path: Path, output_dir: Path) -> None:
    """
    Loads a TensorFlow SavedModel, runs inference on an image, and saves the output.

    Args:
        model_path (Path): Path to the directory containing the tf.saved_model.
        image_path (Path): Path to the input image file.
        output_dir (Path): Directory where the output image will be saved.
    """
    # 1. Load the saved model
    print(f"Loading model from: {model_path}")
    try:
        model = tf.saved_model.load(str(model_path))
    except OSError:
        print(f"Error: Model not found at '{model_path}'.")
        print("Please ensure the model has been trained and saved correctly.")
        return

    # 2. Load and preprocess the input image
    print(f"Loading image: {image_path}")
    if not image_path.is_file():
        print(f"Error: Image file not found at '{image_path}'")
        return

    original_image = cv2.imread(str(image_path))
    if original_image is None:
        print(f"Error: Could not read the image file '{image_path}' via OpenCV.")
        return

    # Preprocess for the model: resize, convert color, normalize to [0, 1]
    img_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (IMG_WIDTH, IMG_HEIGHT))
    img_normalized = img_resized.astype(np.float32) / 255.0

    # Add a batch dimension for the model
    input_tensor = np.expand_dims(img_normalized, axis=0)

    # 3. Run prediction
    print("Running inference...")
    
    # Convert numpy array to a TensorFlow tensor
    input_tensor = tf.constant(input_tensor, dtype=tf.float32)

    # Access the specific signature for inference
    inference_fn = model.signatures['serving_default']
    
    # The output is a dictionary; get the tensor by its output layer name.
    # We dynamically get the first key, assuming there's only one output.
    prediction_dict = inference_fn(input_tensor)
    output_key = list(prediction_dict.keys())[0] 
    prediction_tensor = prediction_dict[output_key]
    
    # Convert tensor to numpy array and squeeze the batch dimension
    prediction_array = prediction_tensor.numpy().squeeze()
    print(f"Model prediction (normalized): {prediction_array}")

    # 4. Post-process: Draw bounding boxes on the original image
    output_image = draw_predicted_boxes(original_image, prediction_array)

    # 5. Save the resulting image
    output_dir.mkdir(parents=True, exist_ok=True)
    output_filename = f"{image_path.stem}_prediction.png"
    output_path = output_dir / output_filename
    cv2.imwrite(str(output_path), output_image)
    print(f"\n✅ Inference complete. Result saved to: {output_path}")

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

