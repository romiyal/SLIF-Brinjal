import os
import time
import logging
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset
import numpy as np
# from torchvision.models import squeezenet1_0, mobilenet_v2
from torchinfo import summary
from calflops import calculate_flops
import torchvision.models as models
# Custom imports
from data_setup import create_dataloaders, print_classwise_counts
from engine_tSNE import train_one_epoch, evaluate_model
from utils import save_model, plot_tsne, save_embeddings
from DilatedBottleneck_Model_n import UltraLightBlockNet_L1

# --- Hyperparameters ---
# data_dir = "/storage/scratch1/m22-pg-plant-pathology/SLIF_Brinjal_Dataset/new_Data/Phase_I/"
#data_dir = "/storage/scratch1/m22-pg-plant-pathology/SLIF_Brinjal_Dataset/new_Data/Phase_I_Augmented/" 
#data_dir = "/storage/scratch1/m22-pg-plant-pathology/SLIF_Brinjal_Dataset/new_Data/Phase_II_AUG_MIX" 
data_dir = "/home2/romiyalg/romiyal/dataset/Plantvillage/PlantVillage-Dataset/raw/color/Tomato/"
#data_dir = "/storage/scratch1/m22-pg-plant-pathology/SLIF_Dataset_work/data/AugMix_final/"
batch_size = 16
input_size = 256 
num_epochs = 500
k_folds = 5
LEARNING_RATE = 1e-4
MIN_LR = 1e-6
PATIENCE = 30     
T_MAX = 500
WEIGHT_DECAY = 0.05
WARMUP_EPOCHS = 10
label_smoothing = 0.001
# CHOOSE YOUR MODEL: "custom", mobilenet_v3_S"mobilenet_v2", "mobilenet_v3", "efficientnet_b0", "shufflenet", "squeezenet"
model_type = "mobilenet_v2" 
model_name = f"Lightweight-{model_type}"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("LEARNING_RATE:", LEARNING_RATE)
print("T_MAX:", T_MAX)
print("MIN_LR:", MIN_LR)
print("Model_Name:", model_name)
print("WEIGHT_DECAY:", WEIGHT_DECAY)
print("Label_Smoothing:", label_smoothing)
print("device:", device)

# --- Directory & Logging Setup ---
timestamp = time.strftime("%Y%m%d-%H%M%S")
save_dir = os.path.join("/storage/scratch1/m22-pg-plant-pathology/SLIF_Brinjal_Dataset/Data/code/output_N6/", f"{model_name}_{timestamp}")
os.makedirs(save_dir, exist_ok=True)

log_filepath = os.path.join(save_dir, f"train_log_{timestamp}.log")
logging.basicConfig(level=logging.INFO, format='%(message)s',
                    handlers=[logging.FileHandler(log_filepath), logging.StreamHandler()])
logger = logging.getLogger()


# --- Data Preparation ---
full_loader = create_dataloaders(data_dir, batch_size, input_size)
# Print original dataset details
print_classwise_counts(full_loader.dataset, full_loader, "Original-Dataset")

train_dataset = full_loader.dataset
targets = [label for _, label in train_dataset.samples]
skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
class_names = train_dataset.classes
print("class_names:", class_names)

num_classes = len(class_names)
best_results_list = []

# --- Model Selection Helper ---
def get_model(m_type, num_classes):
    
    if m_type == "mobilenet_v2":
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(model.last_channel, num_classes)
        return model
    
    elif m_type == "mobilenet_v3_S":
        model = models.mobilenet_v3_small(weights=None)
        last_ch = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(last_ch, num_classes)
        return model
    
    elif m_type == "shufflenet":
        model = models.shufflenet_v2_x0_5(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    
    elif m_type == "squeezenet":
        model = models.squeezenet1_0(weights=None)
        in_channels = model.classifier[1].in_channels  # Get the number of input channels
        model.classifier[1] = nn.Conv2d(in_channels, num_classes, kernel_size=1)
        return model
    
    else:
        raise ValueError(f"Unknown model type: {m_type}")
# --- Helper Function to Print Results Matrix ---
def print_results_matrix(stats, fold_num, best_epoch):
    """Print formatted results matrix for a fold"""
    logger.info("\n" + "="*80)
    logger.info(f"FOLD {fold_num} - BEST EPOCH {best_epoch} RESULTS MATRIX")
    logger.info("="*80)
    logger.info(f"{'Metric':<20} {'Train':<15} {'Validation':<15}")
    logger.info("-"*80)
    logger.info(f"{'Loss':<20} {stats['Train Loss']:<15.4f} {stats['Val Loss']:<15.4f}")
    logger.info(f"{'Accuracy':<20} {stats['Train Acc']:<15.4f} {stats['Val Acc']:<15.4f}")
    logger.info(f"{'Precision':<20} {stats['Train Precision']:<15.4f} {stats['Val Precision']:<15.4f}")
    logger.info(f"{'Recall':<20} {stats['Train Recall']:<15.4f} {stats['Val Recall']:<15.4f}")
    logger.info(f"{'F1-Score':<20} {stats['Train F1']:<15.4f} {stats['Val F1']:<15.4f}")
    logger.info("="*80 + "\n")

# --- Helper Function to Print Average Results Matrix ---
def print_average_results_matrix(results_df):
    """Print formatted average results matrix across all folds"""
    logger.info("\n" + "="*80)
    logger.info("AVERAGE RESULTS ACROSS ALL FOLDS")
    logger.info("="*80)
    logger.info(f"{'Metric':<20} {'Train Mean':<15} {'Train Std':<15} {'Val Mean':<15} {'Val Std':<15}")
    logger.info("-"*80)
    
    metrics = [
        ('Loss', 'Train Loss', 'Val Loss'),
        ('Accuracy', 'Train Acc', 'Val Acc'),
        ('Precision', 'Train Precision', 'Val Precision'),
        ('Recall', 'Train Recall', 'Val Recall'),
        ('F1-Score', 'Train F1', 'Val F1')
    ]
    
    for metric_name, train_col, val_col in metrics:
        train_mean = results_df[train_col].mean()
        train_std = results_df[train_col].std()
        val_mean = results_df[val_col].mean()
        val_std = results_df[val_col].std()
        logger.info(f"{metric_name:<20} {train_mean:<15.4f} {train_std:<15.4f} {val_mean:<15.4f} {val_std:<15.4f}")
    
    logger.info("="*80)
    
    # Print individual fold results
    logger.info("\nINDIVIDUAL FOLD RESULTS:")
    logger.info("-"*80)
    for idx, row in results_df.iterrows():
        logger.info(f"Fold {idx+1}: Train Acc={row['Train Acc']:.4f}, Val Acc={row['Val Acc']:.4f}, "
                   f"Train F1={row['Train F1']:.4f}, Val F1={row['Val F1']:.4f}")
    logger.info("="*80 + "\n")

# --- Helper Function to Plot Training Curves ---
def plot_training_curves(history, save_path, fold_num):
    """Plot training and validation loss and accuracy using matplotlib"""
    import matplotlib.pyplot as plt
    
    epochs = range(1, len(history) + 1)
    
    train_losses = [h["Train Loss"] for h in history]
    val_losses = [h["Val Loss"] for h in history]
    train_accs = [h["Train Acc"] for h in history]
    val_accs = [h["Val Acc"] for h in history]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    axes[0].plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epochs', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title(f'Training and Validation Loss - Fold {fold_num}', fontsize=14)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[1].plot(epochs, train_accs, 'b-', label='Train Accuracy', linewidth=2)
    axes[1].plot(epochs, val_accs, 'r-', label='Val Accuracy', linewidth=2)
    axes[1].set_xlabel('Epochs', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title(f'Training and Validation Accuracy - Fold {fold_num}', fontsize=14)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Training curves saved to {save_path}")

# --- K-Fold Training ---
for fold, (train_indices, val_indices) in enumerate(skf.split(train_dataset.samples, targets)):
    print('\n' + '=' * 30 + f' Fold {fold + 1} ' + '=' * 30)
    base_lr = LEARNING_RATE
    train_loader = DataLoader(Subset(train_dataset, train_indices), batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(Subset(train_dataset, val_indices), batch_size=batch_size, shuffle=False, num_workers=4)
    
    print_classwise_counts(train_dataset, train_loader, "Training")
    print_classwise_counts(train_dataset, val_loader, "Validation")

    model = get_model(model_type, num_classes).to(device)
   
    if fold == 0:
        print(model)
        
        input_shape = (1, 3, input_size, input_size)
        flops, macs, params = calculate_flops(model=model, input_shape=input_shape, output_as_string=True, output_precision=4)
        print("LightBlockNet FLOPs:%s MACs:%s Params:%s \n" % (flops, macs, params))
        summary(model, input_size=(1, 3, input_size, input_size))
    
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=T_MAX, eta_min=MIN_LR)
    scaler = torch.cuda.amp.GradScaler() 

    epochs_no_improve = 0
    best_val_loss = float('inf')
    best_epoch = 0
    fold_history = []
    
    # Best objects for this specific fold
    best_fold_stats = None
    best_val_conf_matrix = None
    # best_embeddings = None
    # best_labels = None

    for epoch in range(num_epochs):
        # Linear Warmup
        print(f"\n----- Epoch {epoch + 1}/{num_epochs} -----")
        if epoch < WARMUP_EPOCHS:
            warmup_lr = base_lr * (epoch + 1) / WARMUP_EPOCHS
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr
            print(f"Epoch {epoch+1}, Warmup Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        else:
            print(f"Epoch {epoch+1}, Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
            scheduler.step()
        
        # Training
        train_loss, train_acc, train_precision, train_recall, train_f1, _ = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
    
        # Evaluation
        # val_loss, val_acc, val_precision, val_recall, val_f1, val_conf, avg_inference_time_ms, val_emb, val_lab = evaluate_model(
        #     model, val_loader, criterion, device, return_embeddings=True
        # )
        val_loss, val_acc, val_precision, val_recall, val_f1, val_conf,avg_inference_time_ms = evaluate_model(
            model, val_loader, criterion, device, return_embeddings=False
        )
        # Log and Print metrics
        metrics_msg = (f"Fold {fold+1} Epoch {epoch+1}/{num_epochs} | "
                       f"Tr-Loss: {train_loss:.4f}, Tr-Acc: {train_acc:.4f}, Tr-Preci: {train_precision:.4f}, "
                       f"Tr-Recall: {train_recall:.4f}, Tr-F1: {train_f1:.4f}, Val-Loss: {val_loss:.4f}, "
                       f"Val-Acc: {val_acc:.4f}, Val-Preci: {val_precision:.4f}, Val-Recall: {val_recall:.4f}, "
                       f"Val-F1: {val_f1:.4f}, Avg Inf Time: {avg_inference_time_ms:.2f} ms")
        logger.info(metrics_msg)

        current_metrics = {
            "Train Loss": train_loss, "Train Acc": train_acc, "Train Precision": train_precision, 
            "Train Recall": train_recall, "Train F1": train_f1, "Val Loss": val_loss, 
            "Val Acc": val_acc, "Val Precision": val_precision, "Val Recall": val_recall, "Val F1": val_f1
        }

        fold_history.append(current_metrics)
        
        # Update Best for Fold
        print("Number of epochs without improvement:", epochs_no_improve)
        if val_loss < best_val_loss:
            print("New best Val_loss:", val_loss)
            best_val_loss = val_loss
            best_epoch = epoch + 1
            epochs_no_improve = 0
            best_fold_stats = current_metrics
            best_val_conf_matrix = val_conf
            # best_embeddings, best_labels = val_emb, val_lab
            save_model(model, save_dir, f"best_{model_name}_fold{fold+1}")
            logger.info(f"*** Best Model Updated (Val Loss: {val_loss:.4f}) ***")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break

    # --- PER-FOLD SAVING ---
    # Plot training curves
    curves_path = os.path.join(save_dir, f"training_curves_fold{fold+1}.png")
    plot_training_curves(fold_history, curves_path, fold+1)
    
    # # Plot t-SNE (using existing function from utils)
    # tsne_path = os.path.join(save_dir, f"tsne_plot_fold{fold+1}.png")
    # plot_tsne(best_embeddings, best_labels, class_names=class_names, save_path=tsne_path)
    
    # # Save embeddings (using existing function from utils)
    # emb_path = os.path.join(save_dir, f"embeddings_fold{fold+1}.pt")
    # save_embeddings(best_embeddings, emb_path)

    # Save confusion matrix
    conf_df = pd.DataFrame(best_val_conf_matrix)
    conf_df.to_csv(os.path.join(save_dir, f"best_val_conf_fold{fold+1}.csv"), index=False)
    
    # Print best epoch results matrix for this fold
    print_results_matrix(best_fold_stats, fold+1, best_epoch)
    
    logger.info(f"Fold {fold+1} results saved (CSV, t-SNE, Embeddings, and Training Curves).")
    best_results_list.append(best_fold_stats)

# --- Final Calculation & Summary ---
results_df = pd.DataFrame(best_results_list)

# Save all fold results to CSV
results_df.to_csv(os.path.join(save_dir, "all_folds_results.csv"), index=False)

# Print average results matrix
print_average_results_matrix(results_df)

# Calculate final averages
avg_train_loss = results_df["Train Loss"].mean()
avg_train_acc = results_df["Train Acc"].mean()
avg_train_perc = results_df["Train Precision"].mean()
avg_train_recall = results_df["Train Recall"].mean()
avg_train_f1 = results_df["Train F1"].mean()

avg_val_loss = results_df["Val Loss"].mean()
avg_val_acc = results_df["Val Acc"].mean()
avg_val_perc = results_df["Val Precision"].mean()
avg_val_recall = results_df["Val Recall"].mean()
avg_val_f1 = results_df["Val F1"].mean()

final_msg = (f"\nFinal Averages | Train Loss: {avg_train_loss:.4f} ± {results_df['Train Loss'].std():.4f} | "
             f"Train Acc: {avg_train_acc:.4f} ± {results_df['Train Acc'].std():.4f} | "
             f"Train Prec: {avg_train_perc:.4f} ± {results_df['Train Precision'].std():.4f} | "
             f"Train Rec: {avg_train_recall:.4f} ± {results_df['Train Recall'].std():.4f} | "
             f"Train F1: {avg_train_f1:.4f} ± {results_df['Train F1'].std():.4f} | "
             f"Val Loss: {avg_val_loss:.4f} ± {results_df['Val Loss'].std():.4f} | "
             f"Val Acc: {avg_val_acc:.4f} ± {results_df['Val Acc'].std():.4f} | "
             f"Val Prec: {avg_val_perc:.4f} ± {results_df['Val Precision'].std():.4f} | "
             f"Val Rec: {avg_val_recall:.4f} ± {results_df['Val Recall'].std():.4f} | "
             f"Val F1: {avg_val_f1:.4f} ± {results_df['Val F1'].std():.4f}")

logger.info("\n" + "="*80)
logger.info("TRAINING COMPLETED SUCCESSFULLY")
logger.info("="*80)
logger.info(final_msg)
logger.info(f"\nAll results saved in: {save_dir}")
logger.info("="*80)