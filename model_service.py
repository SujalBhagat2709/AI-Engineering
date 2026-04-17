import torch
import torch.nn as nn
from torchvision import models
from config import Config

class ModelService:
    
    def __init__(self, model_path=None, num_classes=2):
        
        self.device = Config.get_device()
        
        self.model = models.resnet18(pretrained=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        
        if model_path:
            self.model.load_state_dict(
                torch.load(model_path, map_location=self.device)
            )
        
        self.model.to(self.device)
        self.model.eval()
    
    def predict(self, tensor):
        
        tensor = tensor.to(self.device)
        
        with torch.no_grad():
            output = self.model(tensor)
            pred = output.argmax(dim=1)
        
        return pred.item()