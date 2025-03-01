import torch
import torchvision.transforms as T
from PIL import Image
import os
from torchvision import models
import torch.nn.functional as F  # Import softmax function

def load_model(model_path):
    """Load the PyTorch model from the specified path"""
    # Recreate the same model structure
    model = models.efficientnet_v2_m(weights=None)
    
    # Modify classifier to match the saved model structure
    model.classifier = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Dropout(p=0.5, inplace=True),
        torch.nn.Linear(in_features=1280, out_features=256, bias=True),
        torch.nn.BatchNorm1d(256),
        torch.nn.ReLU(),
        torch.nn.Linear(in_features=256, out_features=2, bias=True)  # 2 classes: AI vs. Human
    )
    
    # Load trained weights
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()  # Set model to evaluation mode
    
    return model

# Define Image Transformations (Same as During Training)
transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_image(image_path, model):
    """Core prediction function that takes a model instance"""
    try:
        image = Image.open(image_path).convert("RGB")  # Load image
        image = transform(image).unsqueeze(0)  # Apply transforms and add batch dimension
    
        with torch.no_grad():  # Disable gradient calculation
            output = model(image)
            probabilities = F.softmax(output, dim=1)  # Convert logits to probabilities
            confidence, prediction = torch.max(probabilities, dim=1)  # Get max probability and class
    
        # Assuming class 0 = Human, class 1 = AI
        class_labels = ["Human", "AI"]
        predicted_label = class_labels[prediction.item()]
        confidence_score = confidence.item()  # Convert tensor to float
    
        return predicted_label, confidence_score
    except Exception as e:
        print(f"Error in predict_image: {e}")
        raise

def predict_single_image(image_path, model_path=None):
    """
    Function to match the interface expected by the app.py
    This is what app.py calls with the uploaded image path
    """
    try:
        # If model path is not provided, use the default one
        if model_path is None or not os.path.exists(model_path):
            model_path = os.path.join(os.path.dirname(__file__), "models", "efficientnet.pth")
        
        # Load the model
        model = load_model(model_path)
        
        # Predict on the image
        label, confidence = predict_image(image_path, model)
        
        return label, confidence
    except Exception as e:
        print(f"Error in predict_single_image: {e}")
        raise

# This section runs only when the script is executed directly
if __name__ == "__main__":
    # Default model path for direct script execution
    default_model_path = "C:/Users/04drm/Projects/aivhumanapp/model-inference-utility/models/efficientnet.pth"
    
    # Test the Model on a New Image
    test_image_path = "C:/Users/04drm/Projects/aivhumanapp/model-inference-utility/test_images/real.jpg"
    
    try:
        label, confidence = predict_single_image(test_image_path, default_model_path)
        print(f"The model predicts this image is: {label} with confidence {confidence:.6f}")
    except Exception as e:
        print(f"Error during testing: {e}")