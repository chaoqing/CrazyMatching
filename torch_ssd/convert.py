import torch
import os
from model import create_ssd_model
from pathlib import Path
import onnx  # Import onnx here as it's used in the function

# Define paths relative to the project root
PYTORCH_MODEL_PATH = "./torch_ssd/ssd_model.pth"
TF_SAVED_MODEL_DIR = "./public/models"
ONNX_MODEL_PATH = os.path.join(TF_SAVED_MODEL_DIR, "crazy_matching.onnx")

example_image = Path(__file__).with_name("example.webp")


def read_torchvision_dummy_input(size=(320, 320)):
    import cv2
    import numpy as np

    image = cv2.imread(example_image)

    smaller_image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
    smaller_image = cv2.cvtColor(smaller_image, cv2.COLOR_BGR2RGB)
    print(f"Resized image shape: {smaller_image.shape}")

    # Add batch dimensios
    input_tensor = (
        np.expand_dims(smaller_image, axis=0).transpose(0, 3, 1, 2).astype(np.float32)
        / 255.0
    )

    return torch.tensor(input_tensor)


def convert_pytorch_to_onnx():
    print("--- Converting PyTorch model to ONNX ---")
    os.makedirs(TF_SAVED_MODEL_DIR, exist_ok=True)

    device = torch.device("cpu")
    model = create_ssd_model(num_classes=16, pretrained=True)
    model.load_state_dict(torch.load(PYTORCH_MODEL_PATH, map_location=device))
    model.eval()
    print(f"PyTorch model loaded from {PYTORCH_MODEL_PATH}")

    # dummy_input = torch.randn(1, 3, 320, 320).to(device)
    w, h = 320, 320  # SSDLite input size
    dummy_input = read_torchvision_dummy_input((w, h)).to(device)

    torch.onnx.export(
        model,
        dummy_input,
        ONNX_MODEL_PATH,
        opset_version=11,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["boxes", "scores", "labels"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "boxes": {0: "num_detections"},
            "scores": {0: "num_detections"},
            "labels": {0: "num_detections"},
        },
    )
    print(f"PyTorch model exported to ONNX at {ONNX_MODEL_PATH}")


def example_onnx_run():
    print("--- Running example ONNX inference ---")
    import onnxruntime as ort
    import cv2
    import numpy as np

    session = ort.InferenceSession(ONNX_MODEL_PATH)
    (input0,) = session.get_inputs()
    w, h = input0.shape[-2:]
    input_name = input0.name

    output_names = [output.name for output in session.get_outputs()]

    image = cv2.imread(example_image)
    print(f"Image shape: {image.shape}")
    img_h, img_w, _ = image.shape

    smaller_image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
    smaller_image = cv2.cvtColor(smaller_image, cv2.COLOR_BGR2RGB)
    print(f"Resized image shape: {smaller_image.shape}")

    # Add batch dimension
    input_tensor = (
        np.expand_dims(smaller_image, axis=0).transpose(0, 3, 1, 2).astype(np.float32)
        / 255.0
    )

    # Run inference
    print(f"Running ONNX inference for {input_tensor.shape}...")
    outputs = session.run(output_names, {input_name: input_tensor})

    print("ONNX model output:")
    for name, output in zip(output_names, outputs):
        print(f"{name}: {output.shape}")
        print(output)

    for i, (boxes, scores, labels) in enumerate(zip(*outputs)):
        xmin, ymin, xmax, ymax = boxes
        xmin *= img_w / w
        ymin *= img_h / h
        xmax *= img_w / w
        ymax *= img_h / h
        color = (0, 255, 0) if i == 0 else (0, 0, 255)
        cv2.rectangle(
            image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, thickness=2
        )
        # box_points = np.intp(box_points)
        # cv2.drawContours(image, [box_points], 0, color, thickness=2)

    cv2.imwrite(example_image.with_suffix(".result.png"), image)


if __name__ == "__main__":
    convert_pytorch_to_onnx()
    example_onnx_run()
