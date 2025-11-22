from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import io
import base64
import os
import traceback

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# RSCD Dataset classes - 27 classes from RSCD dataset
CLASS_NAMES = [
    'dry_asphalt_severe', 'dry_asphalt_slight', 'dry_asphalt_smooth', 
    'dry_concrete_severe', 'dry_concrete_slight', 'dry_concrete_smooth', 
    'dry_gravel', 'dry_mud', 'fresh_snow', 'ice', 'melted_snow', 'water_asphalt_severe', 
    'water_asphalt_slight', 'water_asphalt_smooth', 'water_concrete_severe', 
    'water_concrete_slight', 'water_concrete_smooth', 'water_gravel', 'water_mud', 
    'wet_asphalt_severe', 'wet_asphalt_slight', 'wet_asphalt_smooth', 'wet_concrete_severe', 
    'wet_concrete_slight', 'wet_concrete_smooth', 'wet_gravel', 'wet_mud'
]

# Model configuration
MODEL_PATH = 'C:\\Users\\RAVI KIRAN\\Downloads\\best_model_rscd.pth'  # Full path to your model
NUM_CLASSES = len(CLASS_NAMES)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def create_model_architecture():
    """
    Create ResNet50 with custom FC layer to match your trained model.
    Architecture: ResNet50 backbone + FC(2048->512) + ReLU + Dropout + FC(512->NUM_CLASSES)
    """
    model = models.resnet50(weights=None)
    
    # Replace FC layer with custom architecture matching your trained model
    num_ftrs = model.fc.in_features  # 2048 for ResNet50
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 512),      # fc.0
        nn.ReLU(),                      # fc.1  
        nn.Dropout(0.5),                # fc.2
        nn.Linear(512, NUM_CLASSES)     # fc.3
    )
    return model

def load_model():
    """Load your trained PyTorch model"""
    try:
        print(f"Loading model from: {MODEL_PATH}")
        
        # Load the state_dict
        state_dict = torch.load(MODEL_PATH, map_location=device)
        print(f"✓ Loaded state_dict (type: {type(state_dict)})")
        
        # Create model architecture
        model = create_model_architecture()
        print("✓ Created ResNet50 architecture")
        
        # Load state_dict into model
        model.load_state_dict(state_dict)
        print("✓ Loaded weights into model")
        
        model.to(device)
        model.eval()
        print(f"✓ Model ready on {device}")
        return model
        
    except FileNotFoundError:
        print(f"❌ Model file not found: {MODEL_PATH}")
        print("Please ensure the model file exists in the correct location.")
        return None
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print(traceback.format_exc())
        return None

# Load model
model = load_model()

# Image preprocessing - ADJUST these values if your model used different preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Adjust size as needed
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet defaults
        std=[0.229, 0.224, 0.225]
    )
])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_bytes):
    """Preprocess image for model inference"""
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        return transform(image).unsqueeze(0), image
    except Exception as e:
        raise Exception(f"Error preprocessing image: {str(e)}")

def predict(image_tensor):
    """Run model inference"""
    if model is None:
        raise Exception("Model not loaded. Please check model file and configuration.")
    
    try:
        with torch.no_grad():
            image_tensor = image_tensor.to(device)
            outputs = model(image_tensor)
            
            # Handle different output formats
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            
            probabilities = F.softmax(outputs, dim=1)[0]
            
            # Ensure we have the right number of classes
            if len(probabilities) != len(CLASS_NAMES):
                raise Exception(f"Model outputs {len(probabilities)} classes but expected {len(CLASS_NAMES)}")
            
            # Get all class probabilities
            all_probs = {
                CLASS_NAMES[i]: float(probabilities[i]) * 100 
                for i in range(len(CLASS_NAMES))
            }
            
            # Get top prediction
            confidence, predicted_idx = torch.max(probabilities, 0)
            predicted_class = CLASS_NAMES[predicted_idx.item()]
            
            return predicted_class, float(confidence) * 100, all_probs
            
    except Exception as e:
        raise Exception(f"Error during prediction: {str(e)}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_road_type():
    """API endpoint for road classification"""
    try:
        print("\n=== New prediction request ===")
        
        # Check if model is loaded
        if model is None:
            return jsonify({
                'success': False,
                'error': 'Model not loaded. Please check server logs for details.'
            }), 500
        
        # Check if image file is present
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        print(f"Received file: {file.filename}")
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({
                'error': 'Invalid file type. Please upload PNG, JPG, or JPEG'
            }), 400
        
        # Read and preprocess image
        image_bytes = file.read()
        print(f"Image size: {len(image_bytes)} bytes")
        
        image_tensor, original_image = preprocess_image(image_bytes)
        print(f"Image tensor shape: {image_tensor.shape}")
        
        # Get prediction
        predicted_class, confidence, all_probabilities = predict(image_tensor)
        print(f"Prediction: {predicted_class} ({confidence:.2f}%)")
        
        # Convert image to base64 for displaying
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        
        # Sort probabilities by confidence
        
        sorted_probs = dict(sorted(all_probabilities.items(), 
                                  key=lambda x: x[1], 
                                  reverse=True))
        
        response = {
            'success': True,
            'predicted_class': predicted_class,
            'confidence': round(confidence, 2),
            'all_probabilities': {k: round(v, 2) for k, v in sorted(all_probabilities.items(),key=lambda x: x[1],reverse=True)},
            'image': f"data:image/jpeg;base64,{image_base64}"
        }
        
        print("✓ Prediction successful")
        return jsonify(response)
    
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Error: {error_msg}")
        print(traceback.format_exc())
        
        return jsonify({
            'success': False,
            'error': error_msg
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    model_info = {
        'status': 'healthy',
        'model_loaded': model is not None,
        'device': str(device),
        'num_classes': NUM_CLASSES,
        'class_names': CLASS_NAMES
    }
    
    if model is None:
        model_info['error'] = f'Model file not found at: {MODEL_PATH}'
    
    return jsonify(model_info)

@app.route('/test', methods=['GET'])
def test_endpoint():
    """Test endpoint to verify setup"""
    import sys
    return jsonify({
        'python_version': sys.version,
        'pytorch_version': torch.__version__,
        'model_loaded': model is not None,
        'num_classes': NUM_CLASSES
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)