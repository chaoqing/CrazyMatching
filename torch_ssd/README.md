# PyTorch SSD Object Detection Workflow

This directory contains the necessary scripts to fine-tune an SSD object detection model using PyTorch.

## Setup

1.  **Install Dependencies**:
    ```bash
    uv pip install -r requirements.txt
    ```

## Workflow Steps

1.  **Generate Data**:
    The `generate_tfod_data.py` script (located in `tfod_coco_ssd/data/`) can be used to generate synthetic images and Pascal VOC XML annotations. You might want to copy and adapt it or run it from its original location, specifying the output directory to be within `@torch_ssd`.

    Example (from the project root):
    ```bash
    uv run python tfod_coco_ssd/data/generate_tfod_data.py --num_samples 1000 --output_base_dir torch_ssd/data
    ```
    This will create `images` and `annotations` directories inside `torch_ssd/data`.

2.  **Prepare Dataset and Dataloaders**:
    A custom PyTorch `Dataset` will be implemented to read the generated Pascal VOC data. This will handle parsing XML files and loading images. `DataLoader` will then be used for batching.

3.  **Define and Load SSD Model**:
    A pre-trained SSD model from `torchvision` will be used, and its classification head will be modified to match the 15 custom classes.

4.  **Train the Model**:
    The training script will define the loss function, optimizer, and the training loop.

5.  **Evaluate the Model**:
    An evaluation script will be provided to assess the performance of the trained model on a separate test set.

## File Structure (Planned)

```
torch_ssd/
├── requirements.txt
├── README.md
├── data/
│   ├── images/
│   └── annotations/
├── data_utils.py      # For Dataset and DataLoader
├── model.py           # For SSD model definition
├── train.py           # For training script
└── evaluate.py        # For evaluation script (optional)
```
