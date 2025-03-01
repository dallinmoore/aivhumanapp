import tensorflow as tf
import numpy as np
import torch
from PIL import Image
import os
from transformers import ViTModel
from torchvision import transforms
from tensorflow.keras.layers import Lambda

# ========== EDIT THIS SECTION TO CHANGE THE IMAGE FILE ==========
# Set the path to your image file here
IMAGE_FILE = "C:/Users/04drm/Projects/aivhumanapp/model-inference-utility/test_images/real.jpg"
# Path to model file (change if needed)
MODEL_FILE = "C:/Users/04drm/Projects/aivhumanapp/model-inference-utility/models/combined_model.h5"
# ==================================================================

def predict_single_image(image_path, model_path):
    """
    Use the trained model to predict if an image is AI-generated or not.
    Includes all feature extractors: EfficientNetB0, Xception, and DINO-ViT.
    
    Args:
        image_path: Path to the image file
        model_path: Path to the model file
        
    Returns:
        Prediction result and confidence score
    """
    # Load the DINO-ViT model for feature extraction
    dino_vit = ViTModel.from_pretrained("facebook/dino-vitb8", add_pooling_layer=False)
    dino_vit.eval()  # Set to evaluation mode
    
    # Define the transform for DINO-ViT
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Custom layer for DINO feature extraction
    feature_dim = dino_vit.config.hidden_size  # 768
    
    def extract_dino_features(x):
        x = x.numpy()  # Convert to NumPy
        x = x.transpose(0, 3, 1, 2)  # Convert to (batch, 3, 224, 224)
        x = torch.from_numpy(x).float()
        with torch.no_grad():
            features = dino_vit(x).last_hidden_state[:, 0, :]
        return features.numpy()
    
    def dino_layer(x):
        features = tf.py_function(func=extract_dino_features, inp=[x], Tout=tf.float32)
        features.set_shape((None, feature_dim))
        return features
    
    # Load the trained model
    # We need to provide the custom layer when loading
    custom_objects = {
        'dino_layer': Lambda(dino_layer)
    }
    
    model = tf.keras.models.load_model(model_path, 
                                       custom_objects=custom_objects, 
                                       compile=False)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    
    # Load and preprocess the image for TensorFlow
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    
    # Make prediction
    prediction = model.predict(img_array)
    
    # Get the predicted class and confidence
    predicted_class = np.argmax(prediction[0])
    confidence = prediction[0][predicted_class]
    
    # Map class index to label
    label = "Real" if predicted_class == 0 else "AI-Generated"
    
    return label, confidence

if __name__ == "__main__":
    # Check if the image exists
    if not os.path.exists(IMAGE_FILE):
        print(f"Error: Image file '{IMAGE_FILE}' not found.")
        exit(1)
    
    # Check if model file exists
    if not os.path.exists(MODEL_FILE):
        print(f"Error: Model file '{MODEL_FILE}' not found.")
        exit(1)
    
    # Predict
    try:
        print(f"Analyzing image: {IMAGE_FILE}")
        print("Loading models and processing image... (this may take a moment)")
        label, confidence = predict_single_image(IMAGE_FILE, MODEL_FILE)
        print("\nResults:")
        print(f"Prediction: {label}")
        print(f"Confidence: {confidence:.2%}")
        
        # Provide interpretation
        if confidence > 0.9:
            strength = "very strong"
        elif confidence > 0.75:
            strength = "strong"
        elif confidence > 0.6:
            strength = "moderate"
        else:
            strength = "weak"
            
        print(f"\nThe model has {strength} confidence that this image is {label.lower()}.")
        
        if confidence < 0.6:
            print("Note: This prediction has low confidence and may not be reliable.")
            
    except Exception as e:
        print(f"Error during prediction: {e}")
        import traceback
        traceback.print_exc()  # Print the full stack trace for debugging
