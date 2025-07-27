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
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.layers import Dense, GlobalAveragePooling1D, Input, Reshape, LayerNormalization, MultiHeadAttention
from tensorflow.keras.models import Model
import numpy as np
import os
from sklearn.model_selection import train_test_split

# Define the input image size for the model
# MobileNetV3 typically expects input sizes like (224, 224), (160, 160), etc.
# Adjust this based on your training data and desired performance.
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_CHANNELS = 3

# Number of bounding box coordinates (x, y, width, height, rotation) for each box
# We are predicting 2 bounding boxes, so 2 * 5 = 10 outputs
NUM_BBOX_COORDS = 10 

def load_and_preprocess_data(data_dir):
    """
    Loads pre-generated NumPy data and splits it into training, validation, and test sets.
    """
    X_path = data_dir / 'X_train.npy'
    y_path = data_dir / 'y_train.npy'

    if not X_path.exists() or not y_path.exists():
        raise FileNotFoundError(
            f"Training data not found. Please run `model/data/simulate_data.py` to generate "
            f"'{X_path}' and '{y_path}' first."
        )

    print(f"Loading data from {X_path} and {y_path}...")
    X = np.load(X_path)
    y = np.load(y_path)
    
    # The data from simulate_data is already resized and normalized.
    # We just need to split it.
    
    # Split data into training (80%), validation (10%), and test (10%) sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def create_transformer_encoder_block(embed_dim, num_heads, ff_dim, rate=0.1):
    """Creates a single Transformer Encoder block."""
    inputs = Input(shape=(None, embed_dim))
    
    # Multi-Head Attention layer
    attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(inputs, inputs)
    
    # Add & Norm for the attention output
    out1 = LayerNormalization(epsilon=1e-6)(inputs + attention_output)
    
    # Feed-Forward Network
    ffn_output = Dense(ff_dim, activation="relu")(out1)
    ffn_output = Dense(embed_dim)(ffn_output)

    # Add & Norm for the FFN output
    # The output shape is the same as the input shape
    return Model(inputs=inputs, outputs=LayerNormalization(epsilon=1e-6)(out1 + ffn_output))

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

    # Reshape the feature map into a sequence of patches for the Transformer
    # (7, 7, 576) -> (49, 576)
    h, w, c = x.shape[1:]
    x = Reshape((h * w, c))(x) # Shape: (None, 49, 576)

    # --- Transformer Head Starts Here ---
    # Create and apply a Transformer Encoder block to find relationships between patches
    transformer_block = create_transformer_encoder_block(embed_dim=c, num_heads=8, ff_dim=c*2)
    x = transformer_block(x) # Shape: (None, 49, 576)

    # --- Prediction Head ---
    # Instead of GlobalAveragePooling2D, we pool the sequence from the Transformer.
    # The Transformer has already processed spatial relationships, so pooling now aggregates this rich info.
    x = GlobalAveragePooling1D()(x) # Shape: (None, 576)
    
    # Simple feed-forward network for bounding box prediction
    # The output layer has NUM_BBOX_COORDS (8) units for 2 bounding boxes (x,y,w,h each)
    # Use a linear activation for regression tasks like bounding box prediction.
    x = Dense(NUM_BBOX_COORDS*4, activation='relu')(x)
    x = Dense(NUM_BBOX_COORDS*2, activation='relu')(x)
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
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='mse', metrics=['mae'])

    # 3. Prepare dataset
    data_dir = Path(__file__).parent / 'data' / 'training_data'
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_and_preprocess_data(data_dir)
    
    print(f"\nDataset loaded and split:")
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")

    # 4. Train the model
    print("\nTraining the model with real data...")
    history = model.fit(X_train, y_train, 
                        epochs=100,  # Increased epochs for better convergence
                        batch_size=100, # Smaller batch size for potentially better training
                        validation_data=(X_val, y_val),
                        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)])

    # 5. Evaluate the model on the test set
    print("\nEvaluating model on the test set...")
    test_loss, test_mae = model.evaluate(X_test, y_test)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test MAE: {test_mae:.4f}")

    # 6. Save the trained model
    save_path = Path(__file__).parent/'saved_model'/'my_custom_object_detection_model'
    os.makedirs(save_path, exist_ok=True)
    model.export(str(save_path))
    print(f"\nModel training complete. Model saved to '{save_path}'.")

if __name__ == '__main__':
    # Add scikit-learn to requirements if it's not there
    try:
        import sklearn
    except ImportError:
        print("scikit-learn not found. Please install it using: pip install scikit-learn")
    else:
        train_model()