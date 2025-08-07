import torch
import torch.nn as nn # Import torch.nn
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
    # Load a pre-trained SSDLite320_MobileNet_V3_Large model
    # We use SSDLite320_MobileNet_V3_Large_Weights.DEFAULT to get the best available weights
    model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(weights=torchvision.models.detection.SSDLite320_MobileNet_V3_Large_Weights.DEFAULT if pretrained else None)

    # Retrieve necessary parameters from the original model
    # For SSDLite320_MobileNet_V3_Large, the input size is 320x320
    in_channels = det_utils.retrieve_out_channels(model.backbone, (320, 320))
    num_anchors = model.anchor_generator.num_anchors_per_location()

    # Create new classification and regression heads for SSDLite
    model.head.classification_head = torchvision.models.detection.ssdlite.SSDLiteClassificationHead(in_channels, num_anchors, num_classes, norm_layer=nn.BatchNorm2d)
    model.head.regression_head = torchvision.models.detection.ssdlite.SSDLiteRegressionHead(in_channels, num_anchors, norm_layer=nn.BatchNorm2d)

    return model

if __name__ == '__main__':
    # Example usage:
    model = create_ssd_model(pretrained=True)
    print(model)

    # Test with a dummy input
    dummy_input = torch.randn(1, 3, 320, 320) # Batch size 1, 3 channels, 320x320 image
    model.eval()
    with torch.no_grad():
        predictions, = model(dummy_input)
    
    print("\nModel output:")
    for k, v in predictions.items():
        print(f"{k}: {type(v)} {v.shape} {v.dtype}")
