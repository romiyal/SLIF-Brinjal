import torch
import time
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    conf_matrix = confusion_matrix(all_labels, all_preds)

    return avg_loss, accuracy, precision, recall, f1, conf_matrix

def evaluate_model(model, val_loader, criterion, device, return_embeddings=False):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    embeddings = []
    
    start_time = time.time()
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            if return_embeddings:
                # Assuming the layer before classifier is the embedding
                # This might need adjustment based on model structure
                # For UltraLightBlockNet_L1, we can extract from self.head
                # Let's assume we can get it from a hook or by modifying forward
                # For now, let's keep it simple or use the flattened features
                pass

    end_time = time.time()
    avg_inference_time_ms = ((end_time - start_time) / total) * 1000 if total > 0 else 0

    avg_loss = total_loss / len(val_loader)
    accuracy = correct / total
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    conf_matrix = confusion_matrix(all_labels, all_preds)

    if return_embeddings:
        return avg_loss, accuracy, precision, recall, f1, conf_matrix, torch.tensor([]), [], all_labels, avg_inference_time_ms
    
    return avg_loss, accuracy, precision, recall, f1, conf_matrix, avg_inference_time_ms
