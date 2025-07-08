import os
import sys
import cv2
import torch
from flask import Flask, request, render_template, jsonify
from predict import load_model, preprocess_image, predict
from predict_svm import load_svm_model, preprocess_image_for_svm, predict_svm
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__, template_folder='../templates')
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESNET_MODEL_PATH = os.path.join(BASE_DIR, "models", "resnet18_pneumonia.pth")
SVM_MODEL_PATH = os.path.join(BASE_DIR, "models", "svm_pipeline_model.pkl")

print(f"Chemin de base: {BASE_DIR}")
print(f"Chemin du modèle ResNet: {RESNET_MODEL_PATH}")
print(f"Chemin du modèle SVM: {SVM_MODEL_PATH}")

resnet_model = None
device = None
svm_pipeline = None

def get_resnet_model():
    global resnet_model, device
    if resnet_model is None:
        model, device = load_model(RESNET_MODEL_PATH)
        resnet_model = model
    return resnet_model, device

def get_svm_model():
    global svm_pipeline
    if svm_pipeline is None:
        svm_pipeline = load_svm_model(SVM_MODEL_PATH)
    return svm_pipeline

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_image():
    if 'file' not in request.files:
        return jsonify({'error': 'Aucun fichier envoyé'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Aucun fichier sélectionné'})
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)
    
    model_type = request.form.get('model_type', 'resnet')
    
    try:
        if model_type == 'svm':
            pipeline = get_svm_model()
            processed_image = preprocess_image_for_svm(filepath)
            prediction, confidence = predict_svm(pipeline, processed_image)
        else:
            model, device = get_resnet_model()
            processed_image = preprocess_image(filepath)
            prediction, confidence = predict(model, processed_image, device)
        
        result = {
            'prediction': prediction,
            'confidence': confidence,
            'model_used': model_type
        }
    except Exception as e:
        result = {'error': str(e)}
        print(f"Erreur lors de la prédiction: {e}")
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)