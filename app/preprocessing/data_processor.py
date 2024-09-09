import cv2
import librosa
import torch
from transformers import AutoTokenizer
import numpy as np

class DataProcessor:
    def __init__(self, video_frame_rate=5, audio_duration=10, max_text_length=128):
        self.video_frame_rate = video_frame_rate
        self.audio_duration = audio_duration
        self.max_text_length = max_text_length
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    def extract_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if len(frames) % (30 // self.video_frame_rate) == 0:
                frame = cv2.resize(frame, (224, 224))
                frames.append(frame)
        cap.release()
        return torch.tensor(frames).permute(0, 3, 1, 2) / 255.0

    def extract_audio_features(self, audio_path):
        y, sr = librosa.load(audio_path, duration=self.audio_duration)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        return torch.tensor(mfcc).unsqueeze(0)

    def process_text(self, text):
        return self.tokenizer(text, padding='max_length', truncation=True, 
                              max_length=self.max_text_length, return_tensors="pt")

class MultimodalDataset(torch.utils.data.Dataset):
    def __init__(self, video_paths, audio_paths, texts, labels, processor):
        self.video_paths = video_paths
        self.audio_paths = audio_paths
        self.texts = texts
        self.labels = labels
        self.processor = processor

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        video = self.processor.extract_frames(self.video_paths[idx])
        audio = self.processor.extract_audio_features(self.audio_paths[idx])
        text = self.processor.process_text(self.texts[idx])
        label = torch.tensor(self.labels[idx])
        return video, audio, text, label