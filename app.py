import os
import io
from flask import Flask, request, jsonify
from flask_cors import CORS # For handling Cross-Origin Resource Sharing
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image

# --- Configuration ---
MODEL_STATE_DICT_PATH = 'skin_resnet50_model_statedict.pth' # Should match the saved model name
NUM_CLASSES = 2

# !! IMPORTANT: UPDATE THIS BASED ON YOUR PYTORCH SCRIPT'S LabelEncoder OUTPUT !!
# For example, if your PyTorch script printed: "LabelEncoder mapping: {'diseased': 0, 'healthy': 1}"
# Then CLASS_NAMES should be ['diseased', 'healthy']
# If it printed: "LabelEncoder mapping: {'healthy': 0, 'diseased': 1}"
# Then CLASS_NAMES should be ['healthy', 'diseased']
# This order MUST match the numerical output of your model (0 then 1)
CLASS_NAMES = ['diseased', 'healthy'] # DEFAULT: Example, ADJUST THIS!

# Determine device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Flask App: Using device: {device}")

# --- Model Definition ---
# This MUST be the same architecture as used during training
def get_model(num_classes=NUM_CLASSES):
    # Load a ResNet50 model structure (weights will be loaded from state_dict)
    model = models.resnet50(weights=None) # Important: weights=None as we load our fine-tuned state_dict
    
    # Replace the final fully connected layer (classifier)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

# --- Load Trained Model ---
model = get_model(num_classes=NUM_CLASSES)
try:
    if not os.path.exists(MODEL_STATE_DICT_PATH):
        raise FileNotFoundError(f"Model state_dict not found at {MODEL_STATE_DICT_PATH}. "
                                f"Please ensure the model is saved correctly from your PyTorch training script "
                                f"and this script is in the same directory or the path is correct.")
    
    # Load the state dictionary
    # Handles loading whether model was saved on GPU or CPU
    if str(device) == "cuda" and torch.cuda.is_available():
        model.load_state_dict(torch.load(MODEL_STATE_DICT_PATH))
    else: # If current device is CPU, or if CUDA not available but model might have been saved on GPU
        model.load_state_dict(torch.load(MODEL_STATE_DICT_PATH, map_location=torch.device('cpu')))
    
    model.to(device) # Move model to the determined device
    model.eval()     # Set the model to evaluation mode (important for batchnorm, dropout layers)
    print(f"Model loaded successfully from {MODEL_STATE_DICT_PATH} and set to evaluation mode.")
except Exception as e:
    print(f"Error loading the model: {e}")
    model = None # Ensure model is None if loading failed

# --- Image Preprocessing ---
# These transformations should match what you used during your PyTorch script's
# validation/test phase.
# The updated PyTorch script uses transforms.Resize((224, 224)) for validation.
preprocess = transforms.Compose([
    transforms.Resize((224, 224)), # Match the IMG_SIZE used in PyTorch script's val transform
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Standard ImageNet normalization
])

def transform_image(image_bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        return preprocess(image).unsqueeze(0) # Add batch dimension
    except Exception as e:
        print(f"Error transforming image: {e}")
        return None

# --- Flask App Initialization ---
app = Flask(__name__)
CORS(app) # Enable CORS for all routes

# --- Prediction Endpoint ---
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded or failed to load. Check server logs.'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        img_bytes = file.read()
        tensor = transform_image(img_bytes)
        
        if tensor is None:
            return jsonify({'error': 'Could not process image'}), 400

        tensor = tensor.to(device) # Move tensor to the same device as the model

        with torch.no_grad(): # Disables gradient calculation for inference
            outputs = model(tensor)
            # Outputs are raw scores (logits) for each class
            # Apply softmax to get probabilities
            probabilities = torch.softmax(outputs, dim=1)[0] # Get probabilities for the first image
            
            # Get the predicted class index
            _, predicted_idx_tensor = torch.max(outputs, 1)
            predicted_class_idx = predicted_idx_tensor.item() # Convert tensor to Python number
            
            # Ensure CLASS_NAMES is correctly defined
            if predicted_class_idx >= len(CLASS_NAMES):
                 return jsonify({'error': f'Predicted class index {predicted_class_idx} is out of bounds for CLASS_NAMES. Check CLASS_NAMES in app.py.'}), 500

            predicted_class_name = CLASS_NAMES[predicted_class_idx]
            confidence = probabilities[predicted_class_idx].item() # Confidence of the predicted class

            # Prepare probabilities for all classes
            class_probabilities = {CLASS_NAMES[i]: probabilities[i].item() for i in range(len(CLASS_NAMES))}

        return jsonify({
            'prediction': predicted_class_name,
            'confidence': confidence,
            'class_probabilities': class_probabilities
        })

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': 'Prediction failed. Check server logs.'}), 500

# --- Health Check Endpoint ---
@app.route('/', methods=['GET'])
def health_check():
    return jsonify({'status': 'Healthy', 'message': 'Skin disease prediction API is running.'})

# --- Main ---
if __name__ == '__main__':
    # For development:
    app.run(debug=True, host='0.0.0.0', port=5000)
    # For production, use a proper WSGI server like Gunicorn or Waitress
    # Example with Gunicorn: gunicorn -w 4 -b 0.0.0.0:5000 app:app
