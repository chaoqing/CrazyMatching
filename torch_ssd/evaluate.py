import torch
from torch.utils.data import DataLoader
from data_utils import PascalVOCDataset, get_transform, CLASSES
from model import create_ssd_model
import os
from torchvision.ops import nms
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# Helper function to collate data for DataLoader
def collate_fn(batch):
    return tuple(zip(*batch))

def evaluate_model(model_path, data_root="./data", score_threshold=0.5):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")

    # 1. Load Dataset
    dataset = PascalVOCDataset(os.path.join(data_root), get_transform(train=False))
    data_loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=collate_fn
    )
    print(f"Loaded evaluation dataset with {len(dataset)} samples.")

    # 2. Create Model and Load Weights
    model = create_ssd_model(num_classes=len(CLASSES), pretrained=False) # No pre-trained weights for evaluation
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"Loaded model from {model_path}")

    # 3. Evaluate
    print("Starting evaluation...")
    results = []
    with torch.no_grad():
        for i, (images, targets) in enumerate(data_loader):
            images = list(image.to(device) for image in images)
            outputs = model(images)

            for j, output in enumerate(outputs):
                boxes = output['boxes']
                labels = output['labels']
                scores = output['scores']

                # Apply NMS
                keep = nms(boxes, scores, iou_threshold=0.5)
                boxes = boxes[keep]
                labels = labels[keep]
                scores = scores[keep]

                # Filter by score threshold
                high_scores_idx = torch.where(scores > score_threshold)[0]
                boxes = boxes[high_scores_idx]
                labels = labels[high_scores_idx]
                scores = scores[high_scores_idx]

                results.append({
                    'image_id': targets[j]['image_id'].item(),
                    'boxes': boxes.cpu().numpy(),
                    'labels': labels.cpu().numpy(),
                    'scores': scores.cpu().numpy(),
                    'gt_boxes': targets[j]['boxes'].cpu().numpy(),
                    'gt_labels': targets[j]['labels'].cpu().numpy()
                })
            
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(data_loader)} samples.")

    print("Evaluation complete.")
    
    # Optional: Visualize some predictions
    print("Visualizing a few predictions...")
    num_visualizations = min(5, len(results))
    for k in range(num_visualizations):
        res = results[k]
        img_idx = res['image_id']
        img_name = dataset.image_files[img_idx]
        img_path = os.path.join(dataset.image_dir, img_name)
        img = Image.open(img_path).convert("RGB")

        fig, ax = plt.subplots(1, figsize=(10, 10))
        ax.imshow(img)
        ax.set_title(f"Image: {img_name}")

        # Plot ground truth boxes
        for gt_box, gt_label in zip(res['gt_boxes'], res['gt_labels']):
            xmin, ymin, xmax, ymax = gt_box
            rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, 
                                     linewidth=2, edgecolor='blue', facecolor='none', linestyle='--')
            ax.add_patch(rect)
            ax.text(xmin, ymin - 10, f'GT: {CLASSES[gt_label]}', 
                    bbox=dict(facecolor='blue', alpha=0.7), fontsize=8, color='white')

        # Plot predicted boxes
        for box, label, score in zip(res['boxes'], res['labels'], res['scores']):
            xmin, ymin, xmax, ymax = box
            rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, 
                                     linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
            ax.text(xmin, ymin - 5, f'Pred: {CLASSES[label]} ({score:.2f})', 
                    bbox=dict(facecolor='red', alpha=0.7), fontsize=8, color='white')
        
        plt.axis('off')
        plt.savefig(f"evaluation_example_{img_idx}.png")
        plt.close(fig)
        print(f"Saved evaluation example {img_idx} to evaluation_example_{img_idx}.png")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate a trained SSD object detection model.")
    parser.add_argument('--model_path', type=str, default='./ssd_model.pth', help='Path to the trained model weights.')
    parser.add_argument('--data_root', type=str, default='./data', help='Root directory for dataset.')
    parser.add_argument('--score_threshold', type=float, default=0.5, help='Confidence threshold for predictions.')
    
    args = parser.parse_args()

    model_full_path = os.path.join("./torch_ssd", args.model_path)
    data_full_path = os.path.join("./torch_ssd", args.data_root)

    if not os.path.exists(model_full_path):
        print(f"Error: Model weights not found at {model_full_path}. Please train the model first.")
    elif not os.path.exists(os.path.join(data_full_path, "images")) or \
         not os.path.exists(os.path.join(data_full_path, "annotations")):
        print(f"Error: Data not found in {data_full_path}. Please generate data first.")
    else:
        evaluate_model(model_path=model_full_path,
                       data_root=data_full_path,
                       score_threshold=args.score_threshold)