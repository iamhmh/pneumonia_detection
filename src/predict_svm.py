import os
import sys
import cv2
import numpy as np
import joblib
from PIL import Image

def load_svm_model(model_path):
    """Charge le pipeline SVM (prétraitement + modèle)"""
    try:
        pipeline = joblib.load(model_path)
        print(f"Modèle SVM chargé depuis: {model_path}")
        return pipeline
    except Exception as e:
        print(f"Erreur lors du chargement du modèle SVM: {e}")
        raise

def preprocess_image_for_svm(image_path):
    """Prétraite l'image pour l'adapter au format attendu par le SVM"""
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Impossible de charger l'image: {image_path}")
            
        img_resized = cv2.resize(img, (224, 224))
        
        img_flattened = img_resized.ravel()
        
        img_vector = img_flattened.reshape(1, -1)
        
        return img_vector
    except Exception as e:
        print(f"Erreur lors du prétraitement de l'image: {e}")
        raise

def predict_svm(pipeline, image_vector):
    """Fait une prédiction avec le pipeline SVM"""
    try:
        prediction = pipeline.predict(image_vector)[0]
        
        probabilities = pipeline.predict_proba(image_vector)[0]
        
        print(f"Probabilités brutes: {probabilities}")
        print(f"Classe prédite: {prediction} (0=NORMAL, 1=PNEUMONIA)")
        
        confidence = probabilities[prediction]
        
        classes = ["NORMAL", "PNEUMONIA"]
        return classes[prediction], float(confidence)
    except Exception as e:
        print(f"Erreur lors de la prédiction: {e}")
        raise

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict_svm.py <image_path>")
        sys.exit(1)
    
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    MODEL_PATH = os.path.join(BASE_DIR, "models", "svm_pipeline_model.pkl")
    
    pipeline = load_svm_model(MODEL_PATH)
    
    image_path = sys.argv[1]
    processed_image = preprocess_image_for_svm(image_path)
    
    prediction, confidence = predict_svm(pipeline, processed_image)
    
    print(f"Prédiction SVM: {prediction}")
    print(f"Confiance: {confidence:.2f}")