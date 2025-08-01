import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from data_utils import PascalVOCDataset, get_transform, CLASSES
from model import create_ssd_model
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.ops import box_iou # Import box_iou
from evaluate import evaluate_model # Import the new evaluate_model

# Helper function to collate data for DataLoader
def collate_fn(batch):
    return tuple(zip(*batch))


def train_model(num_epochs=10, batch_size=4, learning_rate=0.001, data_root="./data", save_path="./ssd_model.pth"):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")

    # 1. Load Dataset
    dataset = PascalVOCDataset(os.path.join(data_root), get_transform(train=True))
    data_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=0,
        collate_fn=collate_fn
    )
    print(f"Loaded training dataset with {len(dataset)} samples.")

    # For evaluation, use a separate dataset without data augmentation
    eval_dataset = PascalVOCDataset(os.path.join(data_root), get_transform(train=False))
    print(f"Loaded evaluation dataset with {len(eval_dataset)} samples.")

    # 2. Create Model
    model = create_ssd_model(num_classes=len(CLASSES), pretrained=True)
    model.to(device)
    print("Created SSD model.")

    # 3. Define Optimizer and Learning Rate Scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # 4. Training Loop
    print("Starting training...")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for i, (images, targets) in enumerate(data_loader):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            total_loss += losses.item()

            if (i + 1) % 10 == 0:
                print(f"Epoch: {epoch+1}/{num_epochs}, Batch: {i+1}/{len(data_loader)}, Loss: {losses.item():.4f}")
        
        lr_scheduler.step()
        print(f"Epoch {epoch+1} finished. Average Loss: {total_loss / len(data_loader):.4f}")

        if epoch%5 == 0:
                # Evaluation phase using the new evaluate_model function
                avg_iou = evaluate_model(model, eval_dataset, device, num_visualizations=0) # Set num_visualizations to 0 during training
                print(f"Epoch {epoch+1} Evaluation - Average IoU: {avg_iou:.4f}")

        # 5. Save the trained model
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

    # Optional: Visualize a prediction after training
    model.eval()
    with torch.no_grad():
        # Get one sample from the dataset (without transforms for visualization)
        dataset_eval = PascalVOCDataset(os.path.join(data_root), get_transform(train=False))
        if len(dataset_eval) > 0:
            img_pil, _ = dataset_eval[0] # Get PIL image and original target
            img_tensor = get_transform(train=False)(img_pil, {})[0].unsqueeze(0).to(device)
            prediction = model(img_tensor)[0]

            # Convert tensor to numpy for plotting
            img_np = img_pil.permute(1, 2, 0).cpu().numpy() if isinstance(img_pil, torch.Tensor) else img_pil
            
            fig, ax = plt.subplots(1)
            ax.imshow(img_np)

            for box, label, score in zip(prediction['boxes'], prediction['labels'], prediction['scores']):
                if score > 0.5: # Only show predictions with confidence > 0.5
                    xmin, ymin, xmax, ymax = box.cpu().numpy()
                    rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, 
                                             linewidth=1, edgecolor='r', facecolor='none')
                    ax.add_patch(rect)
                    ax.text(xmin, ymin - 5, f'{CLASSES[label]}: {score:.2f}', 
                            bbox=dict(facecolor='white', alpha=0.7), fontsize=8, color='black')
            plt.axis('off')
            plt.savefig("prediction_example.png")
            print("Saved example prediction to prediction_example.png")
        else:
            print("No samples in the dataset to visualize predictions.")

    return model

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Train an SSD object detection model.")
    parser.add_argument('--num_samples', type=int, default=1000, help='Number of samples to generate for training.')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=50, help='Batch size for training.')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for optimizer.')
    parser.add_argument('--data_root', type=str, default='./data', help='Root directory for dataset.')
    parser.add_argument('--save_path', type=str, default='./ssd_model.pth', help='Path to save the trained model.')
    parser.add_argument('--export_onnx', action='store_true', help='Export the trained model to ONNX format.')
    parser.add_argument('--onnx_output_path', type=str, default='./ssd_model.onnx', help='Path to save the ONNX model.')
    
    args = parser.parse_args()

    # Ensure data is generated before training
    data_gen_script = "./tfod_coco_ssd/data/generate_tfod_data.py"
    data_output_dir = os.path.join("./torch_ssd", args.data_root)
    
    if not os.path.exists(os.path.join(data_output_dir, "images")) or \
       not os.path.exists(os.path.join(data_output_dir, "annotations")):
        print(f"Data not found in {data_output_dir}. Generating {args.num_samples} samples...")
        os.system(f"uv run python {data_gen_script} --num_samples {args.num_samples} --output_base_dir {data_output_dir}")
    
    # Train the model
    train_model(num_epochs=args.num_epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                data_root=data_output_dir,
                save_path=os.path.join("./torch_ssd", args.save_path))

    # Export to ONNX if requested
    if args.export_onnx:
        print(f"Exporting model to ONNX: {os.path.join('./torch_ssd', args.onnx_output_path)}")
        device = torch.device('cpu')
        model = create_ssd_model(num_classes=len(CLASSES), pretrained=True)
        model.load_state_dict(torch.load(os.path.join("./torch_ssd", args.save_path), map_location=device))
        model.to(device)
        model.eval()

        dummy_input = torch.randn(1, 3, 320, 320).to(device) # SSDLite input size
        torch.onnx.export(model, 
                           dummy_input, 
                           args.onnx_output_path,
                           opset_version=11, 
                           input_names=['input'], 
                           output_names=['boxes', 'labels', 'scores'],
                           dynamic_axes={'input' : {0 : 'batch_size'},
                                         'boxes' : {0 : 'num_detections'},
                                         'labels' : {0 : 'num_detections'},
                                         'scores' : {0 : 'num_detections'}})
        print("ONNX model exported successfully.")
