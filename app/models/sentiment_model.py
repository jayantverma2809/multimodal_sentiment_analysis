import torch
import torch.nn as nn
from torchvision import models
from transformers import AutoModel
import torch.quantization

class VideoModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = models.resnet34(pretrained=True)
        self.cnn.fc = nn.Linear(self.cnn.fc.in_features, 128)
        self.lstm = nn.LSTM(128, 64, batch_first=True, bidirectional=True)

    def forward(self, x):
        batch_size, frames, c, h, w = x.shape
        x = x.view(batch_size * frames, c, h, w)
        x = self.cnn(x)
        x = x.view(batch_size, frames, -1)
        x, _ = self.lstm(x)
        return x[:, -1, :]  # Return the last hidden state

class AudioModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(128 * 1 * 1, 128)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class TextModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = AutoModel.from_pretrained("bert-base-uncased")
        self.fc = nn.Linear(768, 128)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.fc(outputs.pooler_output)

class MultimodalAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=4)

    def forward(self, video, audio, text):
        # Combine all modalities
        combined = torch.stack([video, audio, text], dim=0)
        attended, _ = self.attention(combined, combined, combined)
        return attended.mean(dim=0)

class MultimodalSentimentAnalysis(nn.Module):
    def __init__(self):
        super().__init__()
        self.video_module = VideoModule()
        self.audio_module = AudioModule()
        self.text_module = TextModule()
        self.attention = MultimodalAttention(128)
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 3)
        )

    def forward(self, video, audio, text):
        video_features = self.video_module(video)
        audio_features = self.audio_module(audio)
        text_features = self.text_module(text['input_ids'], text['attention_mask'])
        
        fused = self.attention(video_features, audio_features, text_features)
        output = self.classifier(fused)
        return output
    

class QuantizedMultimodalSentimentAnalysis(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.quant = torch.quantization.QuantStub()
        self.model = model
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, video, audio, text):
        video = self.quant(video)
        audio = self.quant(audio)
        output = self.model(video, audio, text)
        return self.dequant(output)

def optimize_for_inference(model):
    model.eval()
    
    # Quantization
    quantized_model = QuantizedMultimodalSentimentAnalysis(model)
    quantized_model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    torch.quantization.prepare(quantized_model, inplace=True)
    torch.quantization.convert(quantized_model, inplace=True)

    # TorchScript
    example_video = torch.randn(1, 20, 3, 224, 224)
    example_audio = torch.randn(1, 1, 13, 100)
    example_text = {'input_ids': torch.randint(0, 1000, (1, 128)), 'attention_mask': torch.ones(1, 128)}
    
    traced_model = torch.jit.trace(quantized_model, (example_video, example_audio, example_text))
    traced_model.save("optimized_model.pt")

    return traced_model