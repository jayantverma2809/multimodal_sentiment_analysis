from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import torch
import io
import aiofiles
import os
from app.models.sentiment_model import MultimodalSentimentAnalysis
from app.preprocessing.data_processor import DataProcessor

app = FastAPI()

model = MultimodalSentimentAnalysis()
model.load_state_dict(torch.load('best_model.pth', map_location=torch.device('cpu')))
model.eval()

processor = DataProcessor()

@app.post("/predict")
async def predict(video: UploadFile = File(...), audio: UploadFile = File(...), text: str = Form(...)):
    try:
        # Save uploaded files temporarily
        video_path = f"temp_{video.filename}"
        audio_path = f"temp_{audio.filename}"
        
        async with aiofiles.open(video_path, 'wb') as out_file:
            content = await video.read()
            await out_file.write(content)
        
        async with aiofiles.open(audio_path, 'wb') as out_file:
            content = await audio.read()
            await out_file.write(content)

        # Process inputs
        video_tensor = processor.extract_frames(video_path)
        audio_tensor = processor.extract_audio_features(audio_path)
        text_encoded = processor.process_text(text)

        # Make prediction
        with torch.no_grad():
            output = model(video_tensor.unsqueeze(0), audio_tensor.unsqueeze(0), text_encoded)
            _, predicted = torch.max(output, 1)

        sentiment_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
        result = sentiment_map[predicted.item()]

        # Clean up temporary files
        os.remove(video_path)
        os.remove(audio_path)

        return JSONResponse(content={'sentiment': result})

    except Exception as e:
        return JSONResponse(content={'error': str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)



from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
import torch
import io
import aiofiles
import os
import numpy as np
from app.models.sentiment_model import MultimodalSentimentAnalysis
from app.preprocessing.data_processor import DataProcessor

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Load the optimized model
model = torch.jit.load("optimized_model.pt")
model.eval()

processor = DataProcessor()

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(request: Request, video: UploadFile = File(...), audio: UploadFile = File(...), text: str = Form(...)):
    try:
        # Save uploaded files temporarily
        video_path = f"temp_{video.filename}"
        audio_path = f"temp_{audio.filename}"
        
        async with aiofiles.open(video_path, 'wb') as out_file:
            content = await video.read()
            await out_file.write(content)
        
        async with aiofiles.open(audio_path, 'wb') as out_file:
            content = await audio.read()
            await out_file.write(content)

        # Process inputs
        video_tensor = processor.extract_frames(video_path)
        audio_tensor = processor.extract_audio_features(audio_path)
        text_encoded = processor.process_text(text)

        # Make prediction
        with torch.no_grad():
            output = model(video_tensor.unsqueeze(0), audio_tensor.unsqueeze(0), text_encoded)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            predicted = torch.argmax(probabilities, dim=1)

        sentiment_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
        result = sentiment_map[predicted.item()]
        confidence = probabilities[0][predicted.item()].item()

        # Clean up temporary files
        os.remove(video_path)
        os.remove(audio_path)

        return templates.TemplateResponse("result.html", {
            "request": request,
            "sentiment": result,
            "confidence": f"{confidence:.2%}"
        })

    except Exception as e:
        return templates.TemplateResponse("error.html", {
            "request": request,
            "error": str(e)
        })

@app.post("/api/predict")
async def api_predict(video: UploadFile = File(...), audio: UploadFile = File(...), text: str = Form(...)):
    try:
        # Save uploaded files temporarily
        video_path = f"temp_{video.filename}"
        audio_path = f"temp_{audio.filename}"
        
        async with aiofiles.open(video_path, 'wb') as out_file:
            content = await video.read()
            await out_file.write(content)
        
        async with aiofiles.open(audio_path, 'wb') as out_file:
            content = await audio.read()
            await out_file.write(content)

        # Process inputs
        video_tensor = processor.extract_frames(video_path)
        audio_tensor = processor.extract_audio_features(audio_path)
        text_encoded = processor.process_text(text)

        # Make prediction
        with torch.no_grad():
            output = model(video_tensor.unsqueeze(0), audio_tensor.unsqueeze(0), text_encoded)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            predicted = torch.argmax(probabilities, dim=1)

        sentiment_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
        result = sentiment_map[predicted.item()]
        confidence = probabilities[0][predicted.item()].item()

        # Clean up temporary files
        os.remove(video_path)
        os.remove(audio_path)

        return JSONResponse(content={
            "sentiment": result,
            "confidence": f"{confidence:.2%}",
            "probabilities": {
                "Negative": f"{probabilities[0][0].item():.2%}",
                "Neutral": f"{probabilities[0][1].item():.2%}",
                "Positive": f"{probabilities[0][2].item():.2%}"
            }
        })

    except Exception as e:
        return JSONResponse(content={'error': str(e)}, status_code=500)

@app.get("/health")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)