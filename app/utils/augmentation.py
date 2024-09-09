import numpy as np
import torch
import cv2
import librosa

class DataAugmenter:
    @staticmethod
    def augment_video(video):
        # Random horizontal flip
        if np.random.rand() > 0.5:
            video = torch.flip(video, [3])
        
        # Random brightness adjustment
        brightness_factor = np.random.uniform(0.8, 1.2)
        video = torch.clamp(video * brightness_factor, 0, 1)
        
        return video

    @staticmethod
    def augment_audio(audio):
        # Time stretching
        stretch_factor = np.random.uniform(0.8, 1.2)
        audio = librosa.effects.time_stretch(audio.numpy().squeeze(), rate=stretch_factor)
        
        # Pitch shifting
        n_steps = np.random.randint(-2, 3)
        audio = librosa.effects.pitch_shift(audio, sr=22050, n_steps=n_steps)
        
        return torch.tensor(audio).unsqueeze(0)

    @staticmethod
    def augment_text(text, tokenizer):
        # Random word deletion
        words = text.split()
        if len(words) > 1:
            delete_idx = np.random.randint(0, len(words))
            words.pop(delete_idx)
            augmented_text = " ".join(words)
        else:
            augmented_text = text
        
        return tokenizer(augmented_text, padding='max_length', truncation=True, 
                         max_length=128, return_tensors="pt")