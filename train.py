import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import KFold
import numpy as np
from tqdm import tqdm
import os
import json
from app.models.sentiment_model import MultimodalSentimentAnalysis
from app.preprocessing.data_processor import MultimodalDataset, DataProcessor
from app.utils.augmentation import DataAugmenter

def reset_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()

def train_model(model, train_loader, val_loader, num_epochs, device, fold=None):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    
    best_val_loss = float('inf')
    patience = 10
    early_stopping_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for video, audio, text, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            video, audio, labels = video.to(device), audio.to(device), labels.to(device)
            text = {k: v.to(device) for k, v in text.items()}

            optimizer.zero_grad()
            outputs = model(video, audio, text)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for video, audio, text, labels in val_loader:
                video, audio, labels = video.to(device), audio.to(device), labels.to(device)
                text = {k: v.to(device) for k, v in text.items()}

                outputs = model(video, audio, text)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        accuracy = 100 * correct / total

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(accuracy)

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Val Accuracy: {accuracy:.2f}%")

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if fold is not None:
                torch.save(model.state_dict(), f'best_model_fold{fold}.pth')
            else:
                torch.save(model.state_dict(), 'best_model.pth')
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= patience:
                print("Early stopping")
                break

    return history

def train_with_cross_validation(model, dataset, num_epochs=50, num_folds=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    
    histories = []

    for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):
        print(f"FOLD {fold+1}")
        print('--------------------------------')

        train_subsampler = SubsetRandomSampler(train_ids)
        val_subsampler = SubsetRandomSampler(val_ids)

        train_loader = DataLoader(dataset, batch_size=32, sampler=train_subsampler)
        val_loader = DataLoader(dataset, batch_size=32, sampler=val_subsampler)

        model.apply(reset_weights)
        model.to(device)

        history = train_model(model, train_loader, val_loader, num_epochs, device, fold)
        histories.append(history)

    # Save the cross-validation results
    with open('cv_results.json', 'w') as f:
        json.dump(histories, f)

    return histories

def prepare_data(video_paths, audio_paths, texts, labels):
    processor = DataProcessor()
    augmenter = DataAugmenter()

    dataset = MultimodalDataset(video_paths, audio_paths, texts, labels, processor, augmenter)
    return dataset

def main():
    # Load your data here
    video_paths = [...]  # List of video file paths
    audio_paths = [...]  # List of audio file paths
    texts = [...]        # List of text transcripts
    labels = [...]       # List of sentiment labels

    dataset = prepare_data(video_paths, audio_paths, texts, labels)
    model = MultimodalSentimentAnalysis()

    # Perform cross-validation
    cv_histories = train_with_cross_validation(model, dataset)

    # Train on the entire dataset
    full_train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    full_val_loader = DataLoader(dataset, batch_size=32, shuffle=False)  # Using the same data for simplicity

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    final_history = train_model(model, full_train_loader, full_val_loader, num_epochs=50, device=device)

    # Save the final training history
    with open('final_training_history.json', 'w') as f:
        json.dump(final_history, f)

    print("Training completed. Model saved as 'best_model.pth'")

if __name__ == "__main__":
    main()