# -*- coding: utf-8 -*-
# Copyright 2025 Nicolas Wang. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from pathlib import Path
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV3Small # Or MobileNetV3Large
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model
import numpy as np
import os

# Define the input image size for the model
# MobileNetV3 typically expects input sizes like (224, 224), (160, 160), etc.
# Adjust this based on your training data and desired performance.
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_CHANNELS = 3

# Number of bounding box coordinates (x, y, width, height, rotation) for each box
# We are predicting 2 bounding boxes, so 2 * 5 = 10 outputs
NUM_BBOX_COORDS = 10 

def create_custom_object_detection_model():
    """
    Creates a custom object detection model using MobileNetV3 as a feature extractor.
    The model predicts two bounding boxes (x, y, width, height) for each.
    """
    input_tensor = Input(shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), name='input_image')

    # Load MobileNetV3 (Small or Large) without the top classification layer
    # Use 'imagenet' weights for pre-training
    base_model = MobileNetV3Small(input_tensor=input_tensor, 
                                  include_top=False, 
                                  weights='imagenet')

    # Freeze the base model layers to prevent them from being updated during initial training
    # This is common for fine-tuning pre-trained models.
    base_model.trainable = False

    # Add a custom head for bounding box prediction
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    
    # Simple feed-forward network for bounding box prediction
    # The output layer has NUM_BBOX_COORDS (8) units for 2 bounding boxes (x,y,w,h each)
    # Use a linear activation for regression tasks like bounding box prediction.
    bbox_output = Dense(NUM_BBOX_COORDS, activation='linear', name='bbox_output')(x)

    model = Model(inputs=input_tensor, outputs=bbox_output, name='custom_object_detector')

    return model

def train_model():
    print("Starting model training...")

    # 1. Create the model
    model = create_custom_object_detection_model()
    model.summary()

    # 2. Compile the model
    # For bounding box regression, Mean Squared Error (MSE) or Huber loss are common.
    # Adam optimizer is a good general-purpose choice.
    model.compile(optimizer='adam', loss='mse') # Using MSE for bounding box regression

    # TODO: 3. Prepare your dataset
    # You will need to load your annotated images and their corresponding bounding box labels.
    # The labels for each image should be a numpy array of shape (8,) representing
    # [x1, y1, w1, h1, x2, y2, w2, h2] for the two target bounding boxes.
    # Ensure your image data is preprocessed (e.g., resized to IMG_HEIGHT, IMG_WIDTH and normalized).
    
    # Example placeholder for dummy data (REPLACE WITH YOUR ACTUAL DATA LOADING)
    num_samples = 100
    dummy_images = np.random.rand(num_samples, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS).astype(np.float32)
    # Dummy bounding box labels (random values between 0 and 1 for normalized coordinates)
    dummy_bboxes = np.random.rand(num_samples, NUM_BBOX_COORDS).astype(np.float32)

    # TODO: 4. Train the model
    # Adjust epochs, batch_size, and add validation_data as needed.
    print("\nTraining the model with dummy data (REPLACE WITH YOUR ACTUAL DATA)...")
    model.fit(dummy_images, dummy_bboxes, epochs=10, batch_size=32)

    # 5. Save the trained model
    # Create the directory if it doesn't exist
    save_path = Path(__file__).parent/'saved_model'/'my_custom_object_detection_model'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.export(save_path) # Exports in TensorFlow SavedModel format
    print(f"\nModel training complete. Model saved to '{save_path}'.")

if __name__ == '__main__':
    train_model()
