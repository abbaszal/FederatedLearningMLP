import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset, random_split
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
MIN_DELTA = 0.001

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
    
    if total == 0: return 0.0, 0.0
    return correct / total, loss_sum / len(loader)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")

    # 1. PREPARE DATA
    print("Loading and preparing data...")
    client_train_datasets = []
    client_test_loaders = []  
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
        
        # Keep a specific loader for this client's test set
        client_test_loaders.append(DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False))
        
        client_sample_counts.append(len(train_ds))

    # Create Global Test and Validation Set
    full_test_set = ConcatDataset(all_test_datasets)
    
    # Split Global Test into Val (for Early Stopping) and Test (for Final Report/Curve)
    val_size = int(0.2 * len(full_test_set))
    test_size = len(full_test_set) - val_size
    global_val_set, global_test_set = random_split(
        full_test_set, [val_size, test_size]
    )

    global_val_loader = DataLoader(global_val_set, batch_size=BATCH_SIZE, shuffle=False)
    global_test_loader = DataLoader(global_test_set, batch_size=BATCH_SIZE, shuffle=False)


    total_samples = sum(client_sample_counts)
    client_weights = [count / total_samples for count in client_sample_counts]

    # 2. INITIALIZE MODEL
    global_model = GaitClassifier(num_classes=NUM_CLASSES).to(device)
    
    best_val_loss = float('inf')
    patience_counter = 0


    global_history = []
    client_history = []

    # 3. FL LOOP
    for round_idx in range(ROUNDS):
        print(f"--- Round {round_idx + 1}/{ROUNDS} ---")
        
        local_models = []

        # A. Local Training & Client Evaluation
        for client_idx, dataset in enumerate(client_train_datasets):

            local_model = copy.deepcopy(global_model)
            local_model.to(device)
            
            loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
            
            # Train locally
            trained_model = train_client(local_model, loader, epochs=LOCAL_EPOCHS)
            local_models.append(trained_model)


            c_acc, c_loss = evaluate(trained_model, global_test_loader)
            
            client_history.append({
                'Round': round_idx + 1,
                'Client_ID': client_idx + 1,
                'Accuracy': c_acc,
                'Loss': c_loss
            })

        # B. Aggregation
        global_model = aggregate_models(global_model, local_models, client_weights)

        # C. Global Evaluation)
        val_acc, val_loss = evaluate(global_model, global_val_loader)
        

        test_acc, test_loss = evaluate(global_model, global_test_loader)
        
        print(f"  Global Val Acc: {val_acc*100:.2f}% | Global Test Acc: {test_acc*100:.2f}%")

   
        global_history.append({
            'Round': round_idx + 1,
            'Val_Accuracy': val_acc,
            'Val_Loss': val_loss,
            'Test_Accuracy': test_acc,
            'Test_Loss': test_loss
        })

        # Early Stopping Logic
        if val_loss < (best_val_loss - MIN_DELTA):
            print(f"  [Improvement] {val_loss:.4f} is better than {best_val_loss:.4f} by at least {MIN_DELTA}")
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(global_model.state_dict(), 'best_global_model.pth')
        else:
            patience_counter += 1
            print(f"  No significant improvement. Patience: {patience_counter}/{PATIENCE}")
            if patience_counter >= PATIENCE:
                print("Early stopping triggered.")
                break

    print("-" * 50)
    df_global = pd.DataFrame(global_history)
    df_global.to_csv('results_global.csv', index=False)

    df_clients = pd.DataFrame(client_history)
    df_clients.to_csv('results_clients.csv', index=False)
    

    global_model.load_state_dict(torch.load('best_global_model.pth'))
    final_test_acc, final_test_loss = evaluate(global_model, global_test_loader)
    print(f"FINAL BEST MODEL TEST ACCURACY: {final_test_acc*100:.2f}%")

if __name__ == "__main__":
    main()