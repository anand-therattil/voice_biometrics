import os 
import sys
import pandas as pd 
import numpy as np 

import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

from cnn import CNN_MODEL


PROJECT_PATH =  os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
print(PROJECT_PATH)

sys.path.append(PROJECT_PATH)
from features.feature_extractor import FeatureExtractor

data_file = pd.read_csv(os.path.join(PROJECT_PATH, 'data.csv'))
print(data_file.head())

class AudioDataset(Dataset):
    def __init__(self, dataframe, feature_extractor):
        self.data = dataframe.reset_index(drop=True)
        self.feature_extractor = feature_extractor
        
        # Map labels to integers: bonafide=0, spoof=1
        self.label_map = {'bonafide': 0, 'spoof': 1}
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        audio_path = row['file_path']
        if self.label_map[row['label']]:
            label = 1
        else:
            label = 0 
        
        # Extract MFCC features
        mfcc = self.feature_extractor.extract_mfcc(audio_path)
        # print(f"{mfcc.shape=}")
        # mfcc shape: (1, 13, time) - already has batch dimension from feature_extractor
        
        return mfcc, label


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        count = 0
        for batch_idx, (mfcc, labels) in enumerate(train_loader):
            mfcc, labels = mfcc.to(device), labels.to(device)
            labels = labels.unsqueeze(0).to(device)
            labels = labels.type(torch.FloatTensor).to(device)
            # print(f"{mfcc.shape=} loader")
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(mfcc)
            # print(f"{outputs}")
            # print(f"{labels}")
            loss = criterion(outputs, labels)
            count += 1

            ACCUMULATION_STEPS = 32
            if (batch_idx + 1) % ACCUMULATION_STEPS == 0:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            else:
                loss.backward() 

            # Statistics
            train_loss += loss.item()
            predicted = (outputs > 0.5).float()
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            if (batch_idx + 1) % 10000 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], '
                      f'Loss: {loss.item():.4f}')
            # print(f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], '
            #           f'Loss: {loss.item():.4f}')
        
        # Calculate training metrics
        train_acc = 100 * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for mfcc, labels in val_loader:
                mfcc, labels = mfcc.to(device), labels.to(device)
                labels = labels.unsqueeze(0).to(device)  # Adds extra dimension
                labels = labels.type(torch.FloatTensor).to(device)
                
                outputs = model(mfcc)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                predicted = (outputs > 0.5).float()
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        
        print(f'\nEpoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%\n')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'Best model saved with validation accuracy: {best_val_acc:.2f}%\n')


# Main execution
if __name__ == "__main__":
    # Hyperparameters
    BATCH_SIZE = 1
    NUM_EPOCHS = 50
    LEARNING_RATE = 0.001
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}\n')
    
    train_data, val_data = train_test_split(
            data_file, 
            test_size=0.2, 
            stratify=data_file['label'],
            random_state=42
        )
    
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    print(f"Label distribution in train: \n{train_data['label'].value_counts()}\n")
    
    # Initialize feature extractor
    feature_extractor = FeatureExtractor()
    
    # Create datasets
    train_dataset = AudioDataset(train_data, feature_extractor)
    val_dataset = AudioDataset(val_data, feature_extractor)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # Set to 0 for debugging, increase for faster loading
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )
    
    # Initialize model
    model = CNN_MODEL().to(device)
    print(f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters\n")
    
    # Loss and optimizer
    criterion =nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # # Optional: Learning rate scheduler
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, mode='min', factor=0.5, patience=5
    # )
    
    # Train the model
    print("Starting training...\n")
    train_model(model, train_loader, val_loader, criterion, optimizer, NUM_EPOCHS, device)
    
    print("\nTraining completed!")
    print(f"Best model saved as 'best_model.pth'")