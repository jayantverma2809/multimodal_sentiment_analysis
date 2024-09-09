import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from app.models.sentiment_model import MultimodalSentimentAnalysis
from app.preprocessing.data_processor import MultimodalDataset, DataProcessor
from app.utils.augmentation import DataAugmenter
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm

def train(model, train_loader, val_loader, num_epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)

    best_val_loss = float('inf')
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

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Val Accuracy: {accuracy:.2f}%")

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')

if __name__ == "__main__":
    # Load your data here
    video_paths = [...]  # List of video file paths
    audio_paths = [...]  # List of audio file paths
    texts = [...]        # List of text transcripts
    labels = [...]       # List of sentiment labels

    # Split the data
    train_video, val_video, train_audio, val_audio, train_texts, val_texts, train_labels, val_labels = train_test_split(
        video_paths, audio_paths, texts, labels, test_size=0.2, random_state=42)

    # Create datasets and dataloaders
    processor = DataProcessor()
    augmenter = DataAugmenter()

    train_dataset = MultimodalDataset(train_video, train_audio, train_texts, train_labels, processor)
    val_dataset = MultimodalDataset(val_video, val_audio, val_texts, val_labels, processor)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = MultimodalSentimentAnalysis()
    train(model, train_loader, val_loader)


from sklearn.model_selection import KFold
import numpy as np

def train_with_cross_validation(model, dataset, num_epochs=10, num_folds=5):
    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    
    for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):
        print(f"FOLD {fold}")
        print('--------------------------------')

        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)

        train_loader = DataLoader(dataset, batch_size=32, sampler=train_subsampler)
        val_loader = DataLoader(dataset, batch_size=32, sampler=val_subsampler)

        model.apply(reset_weights)
        train_model(model, train_loader, val_loader, num_epochs)

def train_model(model, train_loader, val_loader, num_epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    best_val_loss = float('inf')
    patience = 5
    early_stopping_counter = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for video, audio, text, labels in train_loader:
            # ... (training code as before)

        # Validation
        model.eval()
        val_loss = 0.0
        # ... (validation code as before)

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Val Accuracy: {accuracy:.2f}%")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'best_model_fold{fold}.pth')
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= patience:
                print("Early stopping")
                break

def reset_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()

if __name__ == "__main__":
    # ... (load your data here)
    dataset = MultimodalDataset(video_paths, audio_paths, texts, labels, processor)
    model = MultimodalSentimentAnalysis()
    train_with_cross_validation(model, dataset)