import torch
import torchvision
from torchvision.models.detection.ssd import SSD, SSDClassificationHead, SSDRegressionHead
from torchvision.models.detection import _utils as det_utils

# Define the number of classes (15 custom classes + 1 background class)
NUM_CLASSES = 16 # 15 animals + background

def create_ssd_model(num_classes=NUM_CLASSES, pretrained=True):
    """
    Creates a pre-trained SSD300_VGG16 model and modifies its classification head
    to match the specified number of classes.
    """
    # Load a pre-trained SSD300_VGG16 model
    # We use SSD300_VGG16_Weights.DEFAULT to get the best available weights
    model = torchvision.models.detection.ssd300_vgg16(weights=torchvision.models.detection.SSD300_VGG16_Weights.DEFAULT if pretrained else None)

    # Retrieve necessary parameters from the original model
    # For SSD300_VGG16, the input size is 300x300
    in_channels = det_utils.retrieve_out_channels(model.backbone, (300, 300))
    num_anchors = model.anchor_generator.num_anchors_per_location()

    # Create new classification and regression heads
    model.head.classification_head = SSDClassificationHead(in_channels, num_anchors, num_classes)
    model.head.regression_head = SSDRegressionHead(in_channels, num_anchors)

    

    return model

if __name__ == '__main__':
    # Example usage:
    model = create_ssd_model(pretrained=True)
    print(model)

    # Test with a dummy input
    dummy_input = torch.randn(1, 3, 300, 300) # Batch size 1, 3 channels, 300x300 image
    model.eval()
    with torch.no_grad():
        predictions = model(dummy_input)
    
    print("\nModel output (first prediction):")
    print(predictions[0]['boxes'].shape) # Should be [num_detections, 4]
    print(predictions[0]['labels'].shape) # Should be [num_detections]
    print(predictions[0]['scores'].shape) # Should be [num_detections]
