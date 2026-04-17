from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io
from torchvision import transforms
from config import Config
from model_service import ModelService

app = FastAPI()

model_service = ModelService()

transform = transforms.Compose([
    transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
    transforms.ToTensor()
])

@app.get("/")
def health_check():
    return {"status": "API running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    tensor = transform(image).unsqueeze(0)
    
    prediction = model_service.predict(tensor)
    
    return {
        "prediction": int(prediction)
    }