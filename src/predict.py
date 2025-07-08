import os
import sys
import cv2
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms, models
from PIL import Image

def create_model():
    model = models.resnet18(pretrained=False)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 128),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(128, 2)
    )
    return model

def load_model(model_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = create_model()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, device

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    return image

def predict(model, image_tensor, device):
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        prediction = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][prediction].item()
    
    classes = ["NORMAL", "PNEUMONIA"]
    return classes[prediction], confidence

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)
    
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    MODEL_PATH = os.path.join(BASE_DIR, "models", "resnet18_pneumonia.pth")
    
    model, device = load_model(MODEL_PATH)
    
    image_path = sys.argv[1]
    processed_image = preprocess_image(image_path)
    
    prediction, confidence = predict(model, processed_image, device)
    
    print(f"Prédiction: {prediction}")
    print(f"Confiance: {confidence:.2f}")