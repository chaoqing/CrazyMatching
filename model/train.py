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
from tensorflow.keras.layers import Dense, Flatten, Input, Reshape, Layer, MultiHeadAttention, LayerNormalization
from tensorflow.keras.models import Model
import numpy as np
import os
from sklearn.model_selection import train_test_split
import argparse

def parse_opt():
    parser = argparse.ArgumentParser(description="Train a custom object detection model.")
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for phase 1 (feature extraction).')
    parser.add_argument('--fine-tune-epochs', type=int, default=100, help='Number of epochs for phase 2 (fine-tuning).')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for training.')
    parser.add_argument('--learning-rate', type=float, default=0.0001, help='Learning rate for phase 1.')
    parser.add_argument('--fine-tune-learning-rate', type=float, default=0.00001, help='Learning rate for phase 2 (fine-tuning).')
    parser.add_argument('--early-stopping-patience', type=int, default=0, help='Patience for early stopping. Set to 0 to disable.')
    parser.add_argument('--load-weights', action='store_true', help='Load existing model weights if available and continue training.')
    parser.add_argument('--export-model', action='store_true', help='Export the trained model to TensorFlow.js format.')
    return parser.parse_args()

# Define the input image size for the model
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_CHANNELS = 3

# Number of bounding box coordinates (x, y, width, height, rotation) for each box
# We are predicting 2 bounding boxes, so 2 * 5 = 10 outputs
NUM_BBOX_COORDS = 10 

# --- NEW: Custom Metric for Detection Rate ---
class CorrectDetectionRate(tf.keras.metrics.Metric):
    """
    A custom metric to monitor the rate of successful detections.

    A detection is considered "correct" if the predicted bounding box has an
    Intersection over Union (IoU) > 0 with the ground truth box.
    This metric calculates the mean success rate (0%, 50%, or 100%) per sample.

    NOTE: This implementation calculates IoU for axis-aligned bounding boxes
    (ignoring rotation) for computational efficiency during training.
    """
    def __init__(self, name="correct_detection_rate", **kwargs):
        super().__init__(name=name, **kwargs)
        self.total_rate = self.add_weight(name="total_rate", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def _calculate_iou(self, y_true, y_pred):
        # y_true/y_pred shape: (batch, num_boxes, 5), where 5 is [x,y,w,h,r]
        # We only use x, y, w, h for axis-aligned IoU.
        true_box_coords = y_true[..., :4]
        pred_box_coords = y_pred[..., :4]

        # Convert (center_x, center_y, width, height) to (x1, y1, x2, y2)
        true_x1 = true_box_coords[..., 0] - true_box_coords[..., 2] / 2.0
        true_y1 = true_box_coords[..., 1] - true_box_coords[..., 3] / 2.0
        true_x2 = true_box_coords[..., 0] + true_box_coords[..., 2] / 2.0
        true_y2 = true_box_coords[..., 1] + true_box_coords[..., 3] / 2.0

        pred_x1 = pred_box_coords[..., 0] - pred_box_coords[..., 2] / 2.0
        pred_y1 = pred_box_coords[..., 1] - pred_box_coords[..., 3] / 2.0
        pred_x2 = pred_box_coords[..., 0] + pred_box_coords[..., 2] / 2.0
        pred_y2 = pred_box_coords[..., 1] + pred_box_coords[..., 3] / 2.0

        # Calculate intersection coordinates
        inter_x1 = tf.maximum(true_x1, pred_x1)
        inter_y1 = tf.maximum(true_y1, pred_y1)
        inter_x2 = tf.minimum(true_x2, pred_x2)
        inter_y2 = tf.minimum(true_y2, pred_y2)

        # Calculate intersection area
        inter_area = tf.maximum(0.0, inter_x2 - inter_x1) * tf.maximum(0.0, inter_y2 - inter_y1)

        # Calculate union area
        true_area = true_box_coords[..., 2] * true_box_coords[..., 3]
        pred_area = pred_box_coords[..., 2] * pred_box_coords[..., 3]
        union_area = true_area + pred_area - inter_area

        return inter_area / (union_area + 1e-6) # Add epsilon to avoid division by zero

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Reshape from (batch, 10) to (batch, 2, 5) for two boxes
        y_true_reshaped = tf.reshape(y_true, [-1, 2, 5])
        y_pred_reshaped = tf.reshape(y_pred, [-1, 2, 5])

        iou = self._calculate_iou(y_true_reshaped, y_pred_reshaped) # Shape: (batch, 2)
        correct_detections = tf.cast(iou > 0, dtype=tf.float32) # Shape: (batch, 2)
        batch_rate = tf.reduce_mean(correct_detections, axis=1) # Shape: (batch,)

        self.total_rate.assign_add(tf.reduce_sum(batch_rate))
        self.count.assign_add(tf.cast(tf.shape(y_true)[0], dtype=tf.float32))

    def result(self):
        return self.total_rate / self.count

# --- NEW: Positional Encoding Layer ---
class PositionalEncoding(Layer):
    """
    Injects positional information into the input tensor.
    This is crucial for the Transformer to understand spatial relationships.
    """
    def __init__(self, position, d_model):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(position, d_model)

    def get_angles(self, position, i, d_model):
        angles = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return position * angles

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            np.arange(position)[:, np.newaxis],
            np.arange(d_model)[np.newaxis, :],
            d_model,
        )
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        pos_encoding = angle_rads[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, inputs):
        return inputs + self.pos_encoding[:, : tf.shape(inputs)[1], :]

# --- NEW: Self-Contained TransformerEncoder Layer ---
class TransformerEncoder(Layer):
    """
    Custom Transformer Encoder to ensure compatibility across TensorFlow versions.
    It consists of multi-head attention and a feed-forward network.
    """
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention = MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.dense_proj = tf.keras.Sequential(
            [Dense(dense_dim, activation="relu"), Dense(embed_dim)]
        )
        self.layernorm_1 = LayerNormalization(epsilon=1e-6)
        self.layernorm_2 = LayerNormalization(epsilon=1e-6)
        self.supports_masking = True

    def call(self, inputs, mask=None):
        attention_output = self.attention(query=inputs, value=inputs, key=inputs)
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)

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

# --- NEW: Custom Loss Function ---
def create_weighted_mse_loss(location_weight=2.0):
    """
    Creates a custom loss function that penalizes errors in location (x, y) more heavily
    than errors in size (width, height) and angle (rotation).
    """
    def loss(y_true, y_pred):
        # The ground truth and predictions have a shape of (batch_size, 10)
        # Layout: [x1, y1, w1, h1, r1, x2, y2, w2, h2, r2]
        
        # Calculate the standard squared errors
        squared_errors = tf.square(y_true - y_pred)
        
        # Create a weight vector that applies higher weight to x and y coordinates
        # Indices for x,y coordinates are 0, 1 (for box 1) and 5, 6 (for box 2)
        weights = np.ones(NUM_BBOX_COORDS, dtype=np.float32)
        weights[[0, 1, 5, 6]] = location_weight
        
        # Multiply the squared errors by the weights
        weighted_squared_errors = squared_errors * weights
        
        # Return the mean of the weighted errors
        return tf.reduce_mean(weighted_squared_errors)
        
    return loss

# --- UPDATED: Model Creation Function ---
def create_custom_object_detection_model():
    """
    Creates a custom object detection model using MobileNetV3 as a feature extractor.
    The model predicts two bounding boxes (x, y, width, height, rotation) for each.
    """
    input_tensor = Input(shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), name='input_image')

    # Load MobileNetV3 without the top classification layer
    base_model = MobileNetV3Small(input_tensor=input_tensor,
                                  include_top=False,
                                  weights='imagenet')

    # Freeze the base model layers to prevent them from being updated during initial training
    base_model.trainable = False

    # Get the feature map from the backbone
    x = base_model.output

    # Reshape the feature map into a sequence of patches for the Transformer
    # (7, 7, 576) -> (49, 576)
    h, w, c = x.shape[1:]
    x = Reshape((h * w, c))(x) # Shape: (None, 49, 576)

    # --- Transformer Head Starts Here ---
    # 1. Add positional encoding to give the Transformer spatial awareness
    x = PositionalEncoding(h * w, c)(x)

    # 2. Apply a Transformer Encoder block to find relationships between patches
    # Using the built-in Keras layer for simplicity and correctness
    transformer_block = TransformerEncoder(embed_dim=c, num_heads=4, dense_dim=c, name="transformer_encoder")
    x = transformer_block(x)

    # --- Prediction Head ---
    # 3. Flatten the sequence output from the Transformer
    x = Flatten()(x)
    
    # 4. Simple feed-forward network for bounding box prediction
    x = Dense(NUM_BBOX_COORDS*4, activation='relu')(x)
    x = Dense(NUM_BBOX_COORDS*2, activation='relu')(x)
    bbox_output = Dense(NUM_BBOX_COORDS, activation='linear', name='bbox_output')(x)

    model = Model(inputs=input_tensor, outputs=bbox_output, name='custom_object_detector')
    return model, base_model

def train_model():
    opt = parse_opt()
    print("Starting model training...")

    # 3. Prepare dataset
    data_dir = Path(__file__).parent / 'data' / 'training_data'
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_and_preprocess_data(data_dir)
    
    print(f"\nDataset loaded and split:")
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")

    # 1. Create or Load the model
    save_path = Path(__file__).parent/'saved_model'
    if opt.load_weights and os.path.exists(save_path):
        print(f"Loading model from '{save_path}'...")
        # Re-create the custom loss function with the same parameters used during saving
        weighted_mse = create_weighted_mse_loss(location_weight=2.5)
        model = tf.keras.models.load_model(str(save_path), custom_objects={'CorrectDetectionRate': CorrectDetectionRate, 'PositionalEncoding': PositionalEncoding, 'TransformerEncoder': TransformerEncoder, 'loss': weighted_mse})
        base_model = None
        print("Model loaded and base model unfrozen for fine-tuning.")
    else:
        print("Creating a new model...")
        model, base_model = create_custom_object_detection_model()
        # Freeze the base model layers for initial training phase
        base_model.trainable = False

    model.summary()

    # 2. Compile the model with the new custom loss function
    weighted_mse = create_weighted_mse_loss(location_weight=2.5) # Emphasize location accuracy
    
    # Callbacks setup
    callbacks = []
    if opt.early_stopping_patience > 0:
        callbacks.append(tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=opt.early_stopping_patience, restore_best_weights=True))

    # Phase 1: Feature Extraction (only if starting new training or base model is frozen)
    if not opt.load_weights or (base_model is not None and not base_model.trainable):
        print("\nTraining the model (Phase 1: Feature Extraction)...")
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=opt.learning_rate), 
                      loss=weighted_mse, 
                      metrics=['mae', CorrectDetectionRate()])
        history_phase1 = model.fit(X_train, y_train, 
                                   epochs=opt.epochs,
                                   batch_size=opt.batch_size,
                                   validation_data=(X_val, y_val),
                                   callbacks=callbacks)

        # After phase 1, unfreeze the base model for fine-tuning
        base_model.trainable = True
        for l in base_model.layers[:100]:
            l.trainable = False

    # Phase 2: Fine-tuning
    print("\nTraining the model (Phase 2: Fine-tuning)...")
    # Recompile the model with a lower learning rate for fine-tuning
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=opt.fine_tune_learning_rate), 
                  loss=weighted_mse, 
                  metrics=['mae', CorrectDetectionRate()])

    history_phase2 = model.fit(X_train, y_train, 
                               epochs=opt.fine_tune_epochs, 
                               batch_size=opt.batch_size,
                               validation_data=(X_val, y_val),
                               callbacks=callbacks)

    # 5. Evaluate the model on the test set
    print("\nEvaluating model on the test set...")
    test_loss, test_mae, test_cr = model.evaluate(X_test, y_test)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test MAE: {test_mae:.4f}")
    print(f"Test CR: {test_cr:.4f}")

    # 6. Save and/or Export the trained model
    os.makedirs(save_path, exist_ok=True)
    model.save(str(save_path))
    print(f"\nModel saved to '{save_path}'.")

    if opt.export_model:
        exported_path = Path(__file__).parent/'exported_model'
        os.makedirs(exported_path, exist_ok=True)
        model.export(str(exported_path))
        print(f"\nModel exported to '{exported_path}'.")

    if not opt.export_model:
        print("\nModel training complete. No model saved or exported (use --save-model or --export-model).")

if __name__ == '__main__':
    train_model()
