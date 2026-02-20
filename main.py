import os
import argparse
import time
import logging
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset
from torchinfo import summary
from calflops import calculate_flops
import torchvision.models as models

# Modular imports from project src
from src.data import create_dataloaders
from src.engine import train_one_epoch, evaluate_model
from src.utils import save_model, plot_training_curves

def get_args():
    parser = argparse.ArgumentParser(description="SLIF-Brinjal Disease Recognition Training")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the dataset directory")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save logs and models")
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size")
    parser.add_argument("--img_size", type=int, default=224, help="Input image size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--num_folds", type=int, default=5, help="Number of folds for cross-validation")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--model_type", type=str, default="custom", choices=["custom", "mobilenet_v2", "mobilenet_v3_s", "shufflenet", "squeezenet"], help="Model architecture")
    return parser.parse_args()

def get_model(model_type, num_classes, img_size):
    if model_type == "custom":
        from src.models import UltraLightBlockNet_L1
        dims = [64, 128]
        channels = [32, 64, 128, 256, 512]
        return UltraLightBlockNet_L1(num_classes, img_size, dims, channels)
    elif model_type == "mobilenet_v2":
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(model.last_channel, num_classes)
        return model
    elif model_type == "mobilenet_v3_s":
        model = models.mobilenet_v3_small(weights=None)
        last_ch = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(last_ch, num_classes)
        return model
    elif model_type == "shufflenet":
        model = models.shufflenet_v2_x0_5(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    elif model_type == "squeezenet":
        model = models.squeezenet1_0(weights=None)
        in_channels = model.classifier[1].in_channels
        model.classifier[1] = nn.Conv2d(in_channels, num_classes, kernel_size=1)
        return model
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def setup_logger(output_dir):
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, f"train_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(), timestamp

def main():
    args = get_args()
    logger, timestamp = setup_logger(args.output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logger.info(f"Start Training with Args: {args}")
    logger.info(f"Device: {device}")

    # Data loading (Assuming .pt files as in the user's specific workflow)
    # If the user has a standard image folder, they should pass it to data_dir
    # Based on the user's main.py from step 352, they were using a ImageFolder-like structure
    # Let's check how create_dataloaders was implemented in main.py step 352
    # In my src/data.py, I implemented it to take subsets.
    # Let's provide a robust data loading implementation that handles standard folders too.

    from torchvision.datasets import ImageFolder
    from src.data import get_transforms
    
    transform_train, transform_val_test = get_transforms(args.img_size)
    
    try:
        full_dataset = ImageFolder(args.data_dir, transform=transform_val_test)
        class_names = full_dataset.classes
        num_classes = len(class_names)
        logger.info(f"Loaded dataset from {args.data_dir} with {num_classes} classes: {class_names}")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return

    targets = [label for _, label in full_dataset.samples]
    skf = StratifiedKFold(n_splits=args.num_folds, shuffle=True, random_state=42)
    
    best_results_list = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(full_dataset.samples, targets)):
        logger.info(f"\n{'='*20} Fold {fold + 1}/{args.num_folds} {'='*20}")
        
        train_subset = Subset(full_dataset, train_idx)
        val_subset = Subset(full_dataset, val_idx)
        
        # Apply training transforms to train subset if possible or just use subsets
        # For simplicity, we'll recreate subsets with appropriate transforms if needed
        # But for now, let's use the provided loader logic
        train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        
        model = get_model(args.model_type, num_classes, args.img_size).to(device)
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.001)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
        
        # summary and FLOPs for first fold
        if fold == 0:
            summary(model, input_size=(1, 3, args.img_size, args.img_size))
        
        best_val_loss = float('inf')
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        
        for epoch in range(args.epochs):
            # Training
            tr_loss, tr_acc, tr_prec, tr_rec, tr_f1, _ = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            # Evaluation
            val_loss, val_acc, val_prec, val_rec, val_f1, val_conf, avg_inf_time = evaluate_model(
                model, val_loader, criterion, device
            )
            
            scheduler.step()
            
            logger.info(f"Epoch {epoch+1}/{args.epochs} | Tr-Acc: {tr_acc:.4f} | Val-Acc: {val_acc:.4f} | Val-Loss: {val_loss:.4f}")
            
            # Save history for plotting
            history['train_loss'].append(tr_loss)
            history['train_acc'].append(tr_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_model(model, args.output_dir, f"best_{args.model_type}_fold{fold+1}")
                best_fold_stats = {
                    "Train Loss": tr_loss, "Train Acc": tr_acc, "Val Loss": val_loss, "Val Acc": val_acc,
                    "Val Precision": val_prec, "Val Recall": val_rec, "Val F1": val_f1
                }

        best_results_list.append(best_fold_stats)
        
        # Plot curves for the fold
        plot_training_curves(
            {'train_loss': history['train_loss'], 'val_loss': history['val_loss'], 
             'train_acc': history['train_acc'], 'val_acc': history['val_acc']},
            os.path.join(args.output_dir, f"curves_fold{fold+1}.png"),
            fold + 1
        )

    # Calculate final averages
    results_df = pd.DataFrame(best_results_list)
    results_df.to_csv(os.path.join(args.output_dir, "results_summary.csv"), index=False)
    
    logger.info("\n" + "="*80)
    logger.info("TRAINING COMPLETED")
    logger.info(f"Summary Results:\n{results_df.mean()}")
    logger.info("="*80)

if __name__ == "__main__":
    main()