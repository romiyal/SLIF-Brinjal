import os
import torch
import matplotlib.pyplot as plt
import numpy as np

def save_model(model, save_dir, model_name):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{model_name}.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

def plot_training_curves(history, save_path, fold_num):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title(f'Loss Curve - Fold {fold_num}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.title(f'Accuracy Curve - Fold {fold_num}')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def save_embeddings(embeddings_data, save_path):
    torch.save(embeddings_data, save_path)
    print(f"Embeddings saved to {save_path}")

def plot_tsne(embeddings, labels, class_names=None, save_path=None):
    # This would require sklearn.manifold.TSNE
    # Keeping a placeholder if needed
    pass
