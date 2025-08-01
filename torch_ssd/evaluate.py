import torch
from torch.utils.data import DataLoader
from data_utils import PascalVOCDataset, get_transform, CLASSES
from model import create_ssd_model
import os
from torchvision.ops import nms, box_iou # Import box_iou
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# Helper function to collate data for DataLoader
def collate_fn(batch):
    return tuple(zip(*batch))

def evaluate_model(model, dataset, device, score_threshold=0.5, iou_threshold=0.5, num_visualizations=5):
    data_loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=0,
        collate_fn=collate_fn
    )
    model.eval()
    print("Starting evaluation...")
    
    total_iou_sum = 0.0
    total_gt_boxes = 0

    results = [] # To store results for visualization

    with torch.no_grad():
        for i, (images, targets) in enumerate(data_loader):
            images = list(image.to(device) for image in images)
            outputs = model(images)

            for j, output in enumerate(outputs):
                boxes = output['boxes']
                labels = output['labels']
                scores = output['scores']
                gt_boxes = targets[j]['boxes'].to(device)

                # Apply NMS
                keep = nms(boxes, scores, iou_threshold=iou_threshold)
                boxes = boxes[keep]
                labels = labels[keep]
                scores = scores[keep]

                # Filter by score threshold
                high_scores_idx = torch.where(scores > score_threshold)[0]
                boxes = boxes[high_scores_idx]
                labels = labels[high_scores_idx]
                scores = scores[high_scores_idx]

                # Calculate IOU for predicted boxes against all ground truth boxes
                if boxes.shape[0] > 0 and gt_boxes.shape[0] > 0:
                    iou_matrix = box_iou(boxes, gt_boxes)
                    
                    # Sum of max IOU for each predicted box
                    # For each predicted box, find the best matching ground truth box
                    for pred_idx in range(boxes.shape[0]):
                        max_iou_for_pred, _ = torch.max(iou_matrix[pred_idx, :], dim=0)
                        total_iou_sum += max_iou_for_pred.item()
                    
                    total_gt_boxes += gt_boxes.shape[0] # Count ground truth boxes

                results.append({
                    'image_id': targets[j]['image_id'].item(),
                    'boxes': boxes.cpu().numpy(),
                    'labels': labels.cpu().numpy(),
                    'scores': scores.cpu().numpy(),
                    'gt_boxes': targets[j]['boxes'].cpu().numpy(),
                    'gt_labels': targets[j]['labels'].cpu().numpy()
                })
            

    print("Evaluation complete.")
    
    average_iou = total_iou_sum / total_gt_boxes if total_gt_boxes > 0 else 0.0
    print(f"Average IoU (predictions vs. ground truth): {average_iou:.4f}")

    # Optional: Visualize some predictions
    print("Visualizing a few predictions...")
    num_visualizations = min(num_visualizations, len(results))
    for k in range(num_visualizations):
        res = results[k]
        # Assuming data_loader.dataset has image_files attribute
        img_idx = res['image_id']
        img_name = data_loader.dataset.image_files[img_idx]
        img_path = os.path.join(data_loader.dataset.image_dir, img_name)
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
    
    return average_iou

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate a trained SSD object detection model.")
    parser.add_argument('--model_path', type=str, default='./ssd_model.pth', help='Path to the trained model weights.')
    parser.add_argument('--data_root', type=str, default='./data', help='Root directory for dataset.')
    parser.add_argument('--score_threshold', type=float, default=0.5, help='Confidence threshold for predictions.')
    parser.add_argument('--iou_threshold', type=float, default=0.5, help='IoU threshold for NMS and evaluation.')
    parser.add_argument('--num_visualizations', type=int, default=5, help='Number of visualizations to save.')
    
    args = parser.parse_args()

    model_full_path = os.path.join("./torch_ssd", args.model_path)
    data_full_path = os.path.join("./torch_ssd", args.data_root)

    if not os.path.exists(model_full_path):
        print(f"Error: Model weights not found at {model_full_path}. Please train the model first.")
    elif not os.path.exists(os.path.join(data_full_path, "images")) or \
         not os.path.exists(os.path.join(data_full_path, "annotations")):
        print(f"Error: Data not found in {data_full_path}. Please generate data first.")
    else:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        print(f"Using device: {device}")

        # Load Dataset
        dataset = PascalVOCDataset(os.path.join(data_full_path), get_transform(train=False))
        data_loader = DataLoader(
            dataset, batch_size=1, shuffle=False, num_workers=0, # Set num_workers to 0 for Windows compatibility or debugging
            collate_fn=collate_fn
        )
        print(f"Loaded evaluation dataset with {len(dataset)} samples.")

        # Create Model and Load Weights
        model = create_ssd_model(num_classes=len(CLASSES), pretrained=False)
        model.load_state_dict(torch.load(model_full_path, map_location=device))
        model.to(device)
        
        avg_iou = evaluate_model(model=model, 
                                 dataset=dataset, 
                                 device=device, 
                                 score_threshold=args.score_threshold,
                                 iou_threshold=args.iou_threshold,
                                 num_visualizations=args.num_visualizations)
        print(f"Final Average IoU: {avg_iou:.4f}")
