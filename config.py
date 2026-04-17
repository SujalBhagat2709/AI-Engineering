class Config:
    
    MODEL_PATH = "models/model.pth"
    NUM_CLASSES = 2
    
    IMAGE_SIZE = 224
    
    DEVICE = "cuda"
    
    @staticmethod
    def get_device():
        import torch
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")