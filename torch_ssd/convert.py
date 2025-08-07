import torch
import os
from model import create_ssd_model
import onnx # Import onnx here as it's used in the function

# Define paths relative to the project root
PYTORCH_MODEL_PATH = "./torch_ssd/ssd_model.pth"
TF_SAVED_MODEL_DIR = "./tf_saved_model"
ONNX_MODEL_PATH = os.path.join(TF_SAVED_MODEL_DIR, "model.onnx")

def convert_pytorch_to_onnx():
    print("--- Converting PyTorch model to ONNX ---")
    os.makedirs(TF_SAVED_MODEL_DIR, exist_ok=True)

    device = torch.device('cpu')
    model = create_ssd_model(num_classes=16, pretrained=True) # Create model with pretrained backbone for consistent architecture
    model.load_state_dict(torch.load(PYTORCH_MODEL_PATH, map_location=device))
    model.eval()
    print(f"PyTorch model loaded from {PYTORCH_MODEL_PATH}")

    dummy_input = torch.randn(1, 3, 320, 320).to(device)

    torch.onnx.export(model,
                      dummy_input,
                      ONNX_MODEL_PATH,
                      opset_version=11,
                      do_constant_folding=True,
                      input_names=['input'],
                      output_names=['boxes', 'scores', 'labels'],
                      dynamic_axes={'input': {0: 'batch_size'},
                                    'boxes': {0: 'num_detections'},
                                    'scores': {0: 'num_detections'},
                                    'labels': {0: 'num_detections'}
                                    })
    print(f"PyTorch model exported to ONNX at {ONNX_MODEL_PATH}")
if __name__ == '__main__':
    # Ensure onnx is imported for the onnx.load call in the dockerized script
    # This is just for the local script to be aware of onnx, not for the dockerized part
    try:
        import onnx
    except ImportError:
        print("Warning: onnx not found in current environment. It is required for PyTorch to ONNX conversion.")

    convert_pytorch_to_onnx()
