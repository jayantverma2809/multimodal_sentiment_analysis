import torch
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from app.models.sentiment_model import MultimodalSentimentAnalysis
from app.preprocessing.data_processor import MultimodalDataset, DataProcessor
from torch.utils.data import DataLoader

def evaluate(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for video, audio, text, labels in test_loader:
            video, audio, labels = video.to(device), audio.to(device), labels.to(device)
            text = {k: v.to(device) for k, v in text.items()}

            outputs = model(video, audio, text)
            _, predicted = torch.max(outputs.data, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print(classification_report(all_labels, all_preds, target_names=['Negative', 'Neutral', 'Positive']))

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.close()

if __name__ == "__main__":
    # Load your test data here
    test_video_paths = [...]  # List of test video file paths
    test_audio_paths = [...]  # List of test audio file paths
    test_texts = [...]        # List of test text transcripts
    test_labels = [...]       # List of test sentiment labels

    processor = DataProcessor()
    test_dataset = MultimodalDataset(test_video_paths, test_audio_paths, test_texts, test_labels, processor)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = MultimodalSentimentAnalysis()
    model.load_state_dict(torch.load('best_model.pth'))
    
    evaluate(model, test_loader)