import os
from pathlib import Path
import argparse
from ultralytics import YOLO

def main(args):
    # Load a pre-trained YOLOv11n model
    # You can specify a different pre-trained model if needed, e.g., 'yolo11n.pt'
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Pre-trained model not found at {args.model_path}. Please download it first.")
    model = YOLO(args.model_path)

    # Define the path to your custom dataset
    # The dataset should be in YOLO format (images/ and labels/ folders)
    data_dir = os.path.join(args.data_base_dir, 'training_data')
    images_dir = os.path.join(data_dir, 'images')
    labels_dir = os.path.join(data_dir, 'labels')

    # Read the number of classes from classes.txt
    classes_file = os.path.join(data_dir, 'classes.txt')
    if not os.path.exists(classes_file):
        raise FileNotFoundError(f"classes.txt not found at {classes_file}. Please run simulate_data.py with --yolo option first.")

    with open(classes_file, 'r') as f:
        class_names_list = [line.strip() for line in f.readlines()]
    num_classes = len(class_names_list)

    print(f"Number of classes detected: {num_classes}")

    # Update the model's number of classes
    # This will reinitialize the detection head to match the new number of classes
    model.model.yaml['nc'] = num_classes

    # Freeze backbone and neck layers
    # Iterate through model layers and set requires_grad to False for backbone and neck
    # The exact layer names/indices might vary slightly based on YOLOv11n implementation
    # This is a common approach, but you might need to inspect the model structure
    # using `model.info()` or `model.model` to get precise layer names.
    # For simplicity, we'll assume layers up to a certain point are backbone/neck.
    # In ultralytics, you can often freeze by name or by iterating through modules.
    # A more robust way is to load the model and then freeze specific parts.

    # Example of freezing layers (conceptual, adjust based on actual model structure)
    # This part might require more specific knowledge of the YOLOv11n architecture
    # as implemented in ultralytics. For a simple fine-tune of classification head,
    # often the backbone and neck are already pre-trained and you only train the head.
    # Ultralytics `train` function often handles this by default if you specify `freeze` argument.

    # For fine-tuning, we typically only train the detection head.
    # Ultralytics `train` method has a `freeze` argument for this.
    # `freeze=10` means freeze first 10 layers (backbone and part of neck)
    # You might need to adjust this number based on the actual YOLOv11n architecture.
    # A common practice is to freeze all layers except the last detection layers.

    # Prepare data for training
    # Ultralytics expects a data.yaml file or direct paths
    # We'll create a simple data dictionary for direct use
    data_config = Path(__file__).with_name("fine-tune.yaml")
    class_names = '\n  '.join(f'{i}: {name}' for i, name in enumerate(class_names_list))
    data_config.write_text(
f"""
path: {data_dir} # dataset root dir
train: images
val: images
names:
  {class_names}
"""
    )
    # Train the model
    # epochs: Number of training epochs
    # imgsz: Image size for training
    # batch: Batch size
    # name: Name of the training run
    # data: Path to data.yaml or data dictionary
    # freeze: Number of layers to freeze (0 to unfreeze all, -1 to freeze backbone)
    # For fine-tuning classification, we want to freeze most layers except the head.
    # The exact value for `freeze` depends on the model architecture.
    # A common value for freezing backbone is 10 or more, or use `freeze=True` if available for full backbone freeze.
    # Let's try a reasonable number, you might need to adjust this.
    # For YOLOv8 (similar to v11n in structure), `freeze=10` freezes the backbone.
    # For freezing only the classification part, we need to be more precise.
    # Ultralytics `train` function doesn't directly support freezing only classification head.
    # Instead, we freeze the backbone and neck, and the head will be trained.

    print("Starting fine-tuning...")
    results = model.train(
        data=data_config,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch_size,
        name=args.run_name,
        # Freeze backbone and neck. Adjust this value based on YOLOv11n architecture.
        # A value like 10-15 usually covers backbone and initial neck layers.
        # For precise freezing, you might need to manually set `requires_grad=False` for layers.
        # Ultralytics documentation suggests `freeze=10` for backbone.
        freeze=args.freeze_layers,
        # You might want to adjust other parameters like learning rate, optimizer, etc.
        # lr0=0.01, lrf=0.01, optimizer='AdamW'
    )

    print("Fine-tuning complete. Results saved to runs/detect/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv11n model.")
    parser.add_argument(
        '--model_path',
        type=str,
        default='yolo11n.pt', # Path to your pre-trained YOLOv11n weights
        help='Path to the pre-trained YOLOv11n model weights (e.g., yolov11n.pt)'
    )
    parser.add_argument(
        '--data_base_dir',
        type=str,
        default=str(Path(__file__).with_name('data')), # Base directory where training_data is located
        help='Base directory for the dataset (e.g., model/data)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--imgsz',
        type=int,
        default=640,
        help='Image size for training (e.g., 640)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
        help='Batch size for training'
    )
    parser.add_argument(
        '--run_name',
        type=str,
        default='yolov11n_finetune',
        help='Name of the training run (results will be saved in runs/detect/run_name)'
    )
    parser.add_argument(
        '--freeze_layers',
        type=int,
        default=10, # Adjust this based on the actual YOLOv11n architecture
        help='Number of layers to freeze from the beginning of the model. Use 0 to unfreeze all.'
    )

    args = parser.parse_args()
    main(args)
