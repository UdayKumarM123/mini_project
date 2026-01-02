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


UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


CLASS_NAMES = [
    'dry_asphalt_smooth', 
    'dry_asphalt_slight', 
    'dry_asphalt_severe',
    'dry_concrete_smooth', 
    'dry_concrete_slight', 
    'dry_concrete_severe',
    'dry_gravel', 
    'dry_mud', 
    'fresh_snow', 
    'ice', 
    'melted_snow',
    'water_asphalt_smooth', 
    'water_asphalt_slight', 
    'water_asphalt_severe',
    'water_concrete_smooth', 
    'water_concrete_slight', 
    'water_concrete_severe',
    'water_gravel', 
    'water_mud',
    'wet_asphalt_severe', 
    'wet_asphalt_slight', 
    'wet_asphalt_smooth',
    'wet_concrete_severe', 
    'wet_concrete_slight', 
    'wet_concrete_smooth',
    'wet_gravel', 
    'wet_mud'
]


ROAD_DESCRIPTIONS = {
    # DRY ASPHALT
    'dry_asphalt_smooth': {
        'description': 'New/very smooth asphalt, good microtexture, no surface defects.',
        'crr_range': '0.010–0.011',
        'mean_crr': 0.0105,
        'energy_available': 100.00,
        'energy_reduced': 0.00
    },
    'dry_asphalt_slight': {
        'description': 'Slightly worn asphalt, small cracks/texture but generally good.',
        'crr_range': '0.013–0.015',
        'mean_crr': 0.014,
        'energy_available': 75.00,
        'energy_reduced': 25.00
    },
    'dry_asphalt_severe': {
        'description': 'Severely worn/broken asphalt, potholes, rough patches.',
        'crr_range': '0.020–0.030',
        'mean_crr': 0.025,
        'energy_available': 42.00,
        'energy_reduced': 58.00
    },
    
    # DRY CONCRETE
    'dry_concrete_smooth': {
        'description': 'Smooth concrete surface, good finish (e.g., new slab).',
        'crr_range': '0.011–0.013',
        'mean_crr': 0.012,
        'energy_available': 87.50,
        'energy_reduced': 12.50
    },
    'dry_concrete_slight': {
        'description': 'Slightly worn concrete, small joints/texture.',
        'crr_range': '0.013–0.015',
        'mean_crr': 0.014,
        'energy_available': 75.00,
        'energy_reduced': 25.00
    },
    'dry_concrete_severe': {
        'description': 'Rough/damaged concrete (large joints, spalling).',
        'crr_range': '0.020–0.030',
        'mean_crr': 0.025,
        'energy_available': 42.00,
        'energy_reduced': 58.00
    },
    
    # DRY OTHER SURFACES
    'dry_gravel': {
        'description': 'Loose/compacted gravel surface (stones, uneven).',
        'crr_range': '0.020–0.030',
        'mean_crr': 0.025,
        'energy_available': 42.00,
        'energy_reduced': 58.00
    },
    'dry_mud': {
        'description': 'Dry, sticky mud patches that deform under tyre load.',
        'crr_range': '0.100–0.150',
        'mean_crr': 0.125,
        'energy_available': 8.40,
        'energy_reduced': 91.60
    },
    
    # SNOW AND ICE
    'fresh_snow': {
        'description': 'Newly fallen, uncompacted snow layer.',
        'crr_range': '0.030–0.050',
        'mean_crr': 0.040,
        'energy_available': 26.25,
        'energy_reduced': 73.75
    },
    'ice': {
        'description': 'Glazed or hard-packed ice (low surface friction).',
        'crr_range': '0.015–0.030',
        'mean_crr': 0.020,
        'energy_available': 52.50,
        'energy_reduced': 47.50
    },
    'melted_snow': {
        'description': 'Packed/wet snow, slushy layer — high deformation losses.',
        'crr_range': '0.040–0.060',
        'mean_crr': 0.050,
        'energy_available': 21.00,
        'energy_reduced': 79.00
    },
    
    # WATER ON ASPHALT
    'water_asphalt_smooth': {
        'description': 'Thin water film on otherwise smooth asphalt.',
        'crr_range': '0.012–0.014',
        'mean_crr': 0.013,
        'energy_available': 80.77,
        'energy_reduced': 19.23
    },
    'water_asphalt_slight': {
        'description': 'Water pooling in shallow depressions on asphalt.',
        'crr_range': '0.016–0.020',
        'mean_crr': 0.018,
        'energy_available': 58.33,
        'energy_reduced': 41.67
    },
    'water_asphalt_severe': {
        'description': 'Deep standing water / hydroplaning risk on asphalt.',
        'crr_range': '0.025–0.035',
        'mean_crr': 0.030,
        'energy_available': 35.00,
        'energy_reduced': 65.00
    },
    
    # WATER ON CONCRETE
    'water_concrete_smooth': {
        'description': 'Thin water cover on smooth concrete.',
        'crr_range': '0.014–0.016',
        'mean_crr': 0.015,
        'energy_available': 70.00,
        'energy_reduced': 30.00
    },
    'water_concrete_slight': {
        'description': 'Small water patches on concrete surface/joints.',
        'crr_range': '0.016–0.020',
        'mean_crr': 0.018,
        'energy_available': 58.33,
        'energy_reduced': 41.67
    },
    'water_concrete_severe': {
        'description': 'Significant standing water on concrete slabs.',
        'crr_range': '0.030–0.040',
        'mean_crr': 0.035,
        'energy_available': 30.00,
        'energy_reduced': 70.00
    },
    
    # WATER ON OTHER SURFACES
    'water_gravel': {
        'description': 'Gravel with water filling voids — high deformation + drag.',
        'crr_range': '0.030–0.040',
        'mean_crr': 0.035,
        'energy_available': 30.00,
        'energy_reduced': 70.00
    },
    'water_mud': {
        'description': 'Very soft, saturated mud — tyres sink and deform a lot.',
        'crr_range': '0.150–0.250',
        'mean_crr': 0.200,
        'energy_available': 5.25,
        'energy_reduced': 94.75
    },
    
    # WET ASPHALT
    'wet_asphalt_smooth': {
        'description': 'Smooth asphalt with a uniform wet film.',
        'crr_range': '0.012–0.015',
        'mean_crr': 0.0135,
        'energy_available': 77.78,
        'energy_reduced': 22.22
    },
    'wet_asphalt_slight': {
        'description': 'Slightly wet asphalt with some texture.',
        'crr_range': '0.016–0.020',
        'mean_crr': 0.018,
        'energy_available': 58.33,
        'energy_reduced': 41.67
    },
    'wet_asphalt_severe': {
        'description': 'Wet + rough asphalt; combined water and roughness losses.',
        'crr_range': '0.024–0.036',
        'mean_crr': 0.030,
        'energy_available': 35.00,
        'energy_reduced': 65.00
    },
    
    # WET CONCRETE
    'wet_concrete_smooth': {
        'description': 'Smooth concrete with thin water film.',
        'crr_range': '0.014–0.016',
        'mean_crr': 0.015,
        'energy_available': 70.00,
        'energy_reduced': 30.00
    },
    'wet_concrete_slight': {
        'description': 'Slightly wet concrete with minor texture.',
        'crr_range': '0.016–0.020',
        'mean_crr': 0.018,
        'energy_available': 58.33,
        'energy_reduced': 41.67
    },
    'wet_concrete_severe': {
        'description': 'Rough/wet concrete with deep joints and water.',
        'crr_range': '0.030–0.040',
        'mean_crr': 0.035,
        'energy_available': 30.00,
        'energy_reduced': 70.00
    },
    
    # WET OTHER SURFACES
    'wet_gravel': {
        'description': 'Gravel that\'s wet — stones + water produce high losses.',
        'crr_range': '0.030–0.040',
        'mean_crr': 0.035,
        'energy_available': 30.00,
        'energy_reduced': 70.00
    },
    'wet_mud': {
        'description': 'Saturated mud — deep tyre sink and viscous losses.',
        'crr_range': '0.150–0.250',
        'mean_crr': 0.200,
        'energy_available': 5.25,
        'energy_reduced': 94.75
    }
}


MODEL_PATH = 'C:\\Users\\RAVI KIRAN\\Downloads\\best_model_rscd.pth'
NUM_CLASSES = len(CLASS_NAMES)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


def create_model_architecture():
    model = models.resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(nn.Linear(num_ftrs, 512), nn.ReLU(), nn.Dropout(0.5), nn.Linear(512, NUM_CLASSES))
    return model


def load_model():
    try:
        print(f"Loading model from: {MODEL_PATH}")
        state_dict = torch.load(MODEL_PATH, map_location=device)
        model = create_model_architecture()
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        print(f"✓ Model ready on {device}")
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None


model = load_model()
transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(image_bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        return transform(image).unsqueeze(0), image
    except Exception as e:
        raise Exception(f"Error preprocessing image: {str(e)}")


def predict(image_tensor):
    if model is None:
        raise Exception("Model not loaded. Please check model file and configuration.")
    
    try:
        with torch.no_grad():
            image_tensor = image_tensor.to(device)
            outputs = model(image_tensor)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            probabilities = F.softmax(outputs, dim=1)[0]
            if len(probabilities) != len(CLASS_NAMES):
                raise Exception(f"Model outputs {len(probabilities)} classes but expected {len(CLASS_NAMES)}")
            
            all_probs = {}
            for i in range(len(CLASS_NAMES)):
                prob_percentage = float(probabilities[i]) * 100
                if prob_percentage >= 0.01:
                    all_probs[CLASS_NAMES[i]] = round(prob_percentage, 2)
            
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
    try:
        if model is None:
            return jsonify({'success': False, 'error': 'Model not loaded'}), 500
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        file = request.files['image']
        if file.filename == '' or not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file'}), 400
        
        image_bytes = file.read()
        image_tensor, _ = preprocess_image(image_bytes)
        predicted_class, confidence, all_probabilities = predict(image_tensor)
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        sorted_probs = dict(sorted(all_probabilities.items(), key=lambda x: x[1], reverse=True))
        road_info = ROAD_DESCRIPTIONS.get(predicted_class, {})
        mean_crr = road_info.get('mean_crr', 0.004)
        ideal_crr = 0.004
        fuel_increase = ((mean_crr - ideal_crr) / ideal_crr) * 100 if ideal_crr > 0 else 0
        vehicle_mass, distance, gravity = 1500, 100000, 9.81
        energy_loss_mj = (mean_crr * vehicle_mass * gravity * distance) / 1_000_000
        fuel_consumption_liters = energy_loss_mj / 34.2
        co2_emissions = fuel_consumption_liters * 2.3
        safety_rating, safety_color = ("Low", "red") if ("severe" in predicted_class or "mud" in predicted_class or "snow" in predicted_class) else (("Medium", "orange") if ("slight" in predicted_class or "gravel" in predicted_class or "water" in predicted_class) else ("High", "green"))
        
        return jsonify({
            'success': True,
            'predicted_class': predicted_class,
            'confidence': round(confidence, 2),
            'all_probabilities': {k: round(v, 2) for k, v in sorted_probs.items()},
            'image': f"data:image/jpeg;base64,{image_base64}",
            'road_info': {
                'description': road_info.get('description', 'No description available'),
                'crr_range': road_info.get('crr_range', 'N/A'),
                'mean_crr': road_info.get('mean_crr', 0),
                'energy_available': road_info.get('energy_available', 0),
                'energy_reduced': road_info.get('energy_reduced', 0),
                'note': road_info.get('note', ''),
                'calculations': {
                    'fuel_increase_percent': round(fuel_increase, 2),
                    'energy_loss_per_100km': round(energy_loss_mj, 2),
                    'fuel_consumption_100km': round(fuel_consumption_liters, 3),
                    'co2_emissions_100km': round(co2_emissions, 3),
                    'comparison_to_ideal': round(((mean_crr - ideal_crr) / ideal_crr) * 100, 2)
                },
                'safety': {'rating': safety_rating, 'color': safety_color}
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'model_loaded': model is not None, 'device': str(device), 'num_classes': NUM_CLASSES})


@app.route('/test', methods=['GET'])
def test_endpoint():
    import sys
    return jsonify({'python_version': sys.version, 'pytorch_version': torch.__version__, 'model_loaded': model is not None, 'num_classes': NUM_CLASSES})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)