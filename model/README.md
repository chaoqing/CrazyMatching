# Crazy Matching - Custom Object Detection Model

This directory contains the Python project for training a custom object detection model for the Crazy Matching game.

## Purpose

The goal of this project is to train a TensorFlow model that can accurately identify the unique game symbols (e.g., cow, zebra, star) from images. This model will then be converted to TensorFlow.js format for use in the web-based augmented reality application.

## Contents

- `requirements.txt`: Lists the Python dependencies required for model training.
- `train.py`: The main script for data loading, model definition, training, and saving the trained model.
- `data/`: (To be created by user) This directory will store your annotated image datasets.
- `saved_model/`: (To be created by `train.py` after training) This directory will contain the trained TensorFlow SavedModel.

## Setup and Usage

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Data Preparation:**
    *   Collect images of your custom game symbols.
    *   Annotate these images using a tool like `LabelImg` to create bounding box annotations and class labels.
    *   Organize your annotated data (e.g., into `data/train` and `data/val` subdirectories).

3.  **Implement Training Logic:**
    *   Edit `train.py` to implement the data loading, preprocessing, model definition, and training loop. You will likely fine-tune a pre-trained object detection model (e.g., SSD-MobileNet).

4.  **Train the Model:**
    ```bash
    python train.py
    ```

5.  **Model Export:**
    *   After successful training, the `train.py` script should save the model in TensorFlow SavedModel format within the `saved_model/` directory. This model will then be converted to TensorFlow.js format.
