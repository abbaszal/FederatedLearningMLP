import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
import pandas as pd
import numpy as np
import copy

from utils.models import GaitClassifier
from utils.fed_utils import HuGaDataset, aggregate_models


METADATA_DIR = 'metadata'
NUM_CLIENTS = 18
ROUNDS = 20
LOCAL_EPOCHS = 5
BATCH_SIZE = 64
LR = 0.001
PATIENCE = 3  


LABEL_MAP = {
    'Walking': 0, 'Going up': 1, 'Going down': 2, 'Sitting': 3,
    'Sitting down': 4, 'Standing up': 5, 'Standing': 6
}
NUM_CLASSES = len(LABEL_MAP)


def train_client(model, train_loader, epochs):
    model.train()
    device = next(model.parameters()).device 
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    for _ in range(epochs):
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    return model



def evaluate(model, loader):
    model.eval()
    device = next(model.parameters()).device
    
    correct = 0
    total = 0
    loss_sum = 0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss_sum += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    return correct / total, loss_sum / len(loader)



def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")

    # 1. PREPARE DATA
    print("Loading and preparing data...")
    client_train_datasets = []
    all_test_datasets = []
    client_sample_counts = []

    # Load each client's data
    for i in range(1, NUM_CLIENTS + 1):
        train_path = os.path.join(METADATA_DIR, f'train_{i:02d}.csv')
        test_path = os.path.join(METADATA_DIR, f'test_{i:02d}.csv')
        
        if not os.path.exists(train_path): continue

        # Create Datasets
        train_ds = HuGaDataset(csv_path=train_path, label_map=LABEL_MAP)
        test_ds = HuGaDataset(csv_path=test_path, label_map=LABEL_MAP)
        
        client_train_datasets.append(train_ds)
        all_test_datasets.append(test_ds)
        client_sample_counts.append(len(train_ds))

    # Create Global Test and Validation Set
    full_test_set = ConcatDataset(all_test_datasets)
    
    # Split Global Test into Val (for Early Stopping) and Test (for Final Report)
    val_size = int(0.2 * len(full_test_set))
    test_size = len(full_test_set) - val_size
    global_val_set, global_test_set = torch.utils.data.random_split(
        full_test_set, [val_size, test_size]
    )

    global_val_loader = DataLoader(global_val_set, batch_size=BATCH_SIZE, shuffle=False)
    global_test_loader = DataLoader(global_test_set, batch_size=BATCH_SIZE, shuffle=False)

    # Calculate FedAvg Weights
    total_samples = sum(client_sample_counts)
    client_weights = [count / total_samples for count in client_sample_counts]

    # 2. INITIALIZE MODEL
    global_model = GaitClassifier(num_classes=NUM_CLASSES).to(device)
    
    best_val_loss = float('inf')
    patience_counter = 0

    # 3. FL LOOP
    for round_idx in range(ROUNDS):
        print(f"--- Round {round_idx + 1}/{ROUNDS} ---")
        
        local_models = []

        # A. Local Training
        for client_idx, dataset in enumerate(client_train_datasets):
            # Copy global model to client
            local_model = copy.deepcopy(global_model)
            local_model.to(device)
            
            loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
            
            # Train locally
            trained_model = train_client(local_model, loader, epochs=LOCAL_EPOCHS)
            local_models.append(trained_model)

        # B. Aggregation
        global_model = aggregate_models(global_model, local_models, client_weights)

        val_acc, val_loss = evaluate(global_model, global_val_loader)
        
        print(f"Global Validation Accuracy: {val_acc*100:.2f}% | Loss: {val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(global_model.state_dict(), 'best_global_model.pth')
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{PATIENCE}")
            if patience_counter >= PATIENCE:
                print("Early stopping triggered.")
                break

    print("-" * 50)
    global_model.load_state_dict(torch.load('best_global_model.pth'))
    test_acc, test_loss = evaluate(global_model, global_test_loader)
    print(f"FINAL GLOBAL TEST ACCURACY: {test_acc*100:.2f}%")

if __name__ == "__main__":
    main()