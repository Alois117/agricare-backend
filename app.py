from flask import Flask, request, jsonify
import os
import numpy as np
import cv2
import requests
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import logging
from werkzeug.utils import secure_filename
from datetime import datetime
from flask_cors import CORS
from dotenv import load_dotenv 

# Load environment variables from .env
load_dotenv()

# Disable unnecessary logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Constants
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model and labels
model = load_model("plant_disease_model.h5")
with open("classes.txt", "r") as f:
    CATEGORIES = [line.strip() for line in f.readlines()]  

# Load Plant.id API Key securely from .env
PLANT_ID_API_KEY = os.getenv("PLANT_ID_API_KEY")  
PLANT_ID_API_URL = "https://plant.id/api/v3/identification"

# Confidence threshold for overriding predictions
CONFIDENCE_THRESHOLD = 75.0  

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "*"}})
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Local fallback treatment data
TREATMENTS = {
    "Pepper__bell___Bacterial_spot": {
        "scientific_name": "Xanthomonas campestris pv. vesicatoria",
        "organic": "Apply copper-based bactericides weekly. Remove infected leaves.",
        "chemical": "Streptomycin sulfate sprays",
        "prevention": "Use disease-free seeds and rotate crops every 2 years",
        "sources": ["PlantVillage Dataset", "Cornell Cooperative Extension"]
    },
    "Tomato_Early_Blight": {
        "scientific_name": "Alternaria solani",
        "organic": "Copper fungicides + neem oil applications",
        "chemical": "Chlorothalonil or mancozeb fungicides",
        "prevention": "Avoid overhead watering, stake plants for air circulation",
        "sources": ["PlantVillage API", "University of Minnesota Extension"]
    },
    "default": {
        "organic": "Remove infected plant parts. Improve air circulation.",
        "chemical": "Consult local agricultural extension",
        "prevention": "Sanitize tools between plants"
    }
}

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (64, 64))
    img = img.astype('float32') / 255.0
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    return img

def predict_disease(image_path):
    img = preprocess_image(image_path)
    prediction = model.predict(img)
    index = np.argmax(prediction)
    label = CATEGORIES[index]
    confidence = round(float(np.max(prediction)) * 100, 2)
    print(f"\n[Model] Prediction: {label} ({confidence}%)")
    return label, confidence

def get_plant_id_prediction(image_path):
    try:
        with open(image_path, 'rb') as img_file:
            response = requests.post(
                PLANT_ID_API_URL,
                files=[
                    ('images', (os.path.basename(image_path), img_file.read())),
                    ('health', (None, 'all')),
                    ('similar_images', (None, 'true'))
                ],
                headers={'Api-Key': PLANT_ID_API_KEY},
                timeout=20
            )
        if response.status_code in [200, 201]:
            return response.json()
        else:
            print(f"API Error {response.status_code}: {response.text}")
            return None
    except Exception as e:
        print(f"API Call Failed: {str(e)}")
        return None

def get_treatment_from_plantid(access_token):
    try:
        details_url = f"{PLANT_ID_API_URL}/{access_token}?details=local_name,description,treatment"
        response = requests.get(
            details_url,
            headers={"Api-Key": PLANT_ID_API_KEY},
            timeout=10
        )
        if response.status_code == 200:
            details = response.json()
            if 'result' in details and 'disease' in details['result']:
                disease = details['result']['disease']
                treatment = disease.get('treatment', {})
                return {
                    "organic_treatment": treatment.get('organic', ['Not specified'])[0],
                    "chemical_treatment": treatment.get('chemical', ['Not specified'])[0],
                    "prevention": treatment.get('prevention', ['Not specified'])[0],
                    "sources": ["Plant.id API"]
                }
    except Exception as e:
        print(f"[Plant.id Details] Error: {str(e)}")
    return None

def get_treatment(disease_key):
    disease_key = disease_key.strip()
    treatment = TREATMENTS.get(disease_key, TREATMENTS["default"])
    display_name = disease_key.replace("__", " ").replace("_", " ").title()
    return {
        "disease_name": display_name,
        "scientific_name": treatment.get("scientific_name", "Unknown"),
        "organic_treatment": treatment["organic"],
        "chemical_treatment": treatment.get("chemical", ""),
        "prevention": treatment["prevention"],
        "sources": treatment.get("sources", [])
    }

def determine_best_prediction(your_pred, your_conf, plantid_pred, plantid_conf):
    if not plantid_pred:
        return your_pred, your_conf, "your_model"
    if not your_pred:
        return plantid_pred, plantid_conf, "plantid_api"
    if your_conf >= CONFIDENCE_THRESHOLD and plantid_conf < CONFIDENCE_THRESHOLD:
        return your_pred, your_conf, "your_model"
    if plantid_conf >= CONFIDENCE_THRESHOLD and your_conf < CONFIDENCE_THRESHOLD:
        return plantid_pred, plantid_conf, "plantid_api"
    if your_conf >= plantid_conf:
        return your_pred, your_conf, "your_model"
    else:
        return plantid_pred, plantid_conf, "plantid_api"

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(file_path)

    try:
        your_label, your_confidence = predict_disease(file_path)
        plantid_response = get_plant_id_prediction(file_path)

        plantid_disease_name = None
        plantid_disease_confidence = 0.0
        plantid_plant_name = None
        plantid_plant_confidence = 0.0

        if plantid_response and 'result' in plantid_response:
            result = plantid_response['result']
            if 'classification' in result and result['classification']['suggestions']:
                plant_suggestion = result['classification']['suggestions'][0]
                plantid_plant_name = plant_suggestion['name']
                plantid_plant_confidence = round(plant_suggestion['probability'] * 100, 2)
            if 'disease' in result and result['disease']['suggestions']:
                disease_suggestion = result['disease']['suggestions'][0]
                plantid_disease_name = disease_suggestion['name']
                plantid_disease_confidence = round(disease_suggestion['probability'] * 100, 2)

        best_disease, best_confidence, source = determine_best_prediction(
            your_label, your_confidence, plantid_disease_name, plantid_disease_confidence
        )

        if source.startswith("plantid"):
            treatment = get_treatment_from_plantid(plantid_response['access_token']) or get_treatment(best_disease)
        else:
            treatment = get_treatment(best_disease)

        response = {
            "timestamp": datetime.now().isoformat(),
            "final_prediction": treatment["disease_name"],
            "final_confidence": f"{best_confidence}%",
            "prediction_source": source,
            "your_prediction": your_label,
            "your_confidence": f"{your_confidence}%",
            "plantid_prediction": plantid_disease_name,
            "plantid_confidence": f"{plantid_disease_confidence}%" if plantid_disease_name else "N/A",
            "plant_name": plantid_plant_name or "Unknown",
            "plant_confidence": f"{plantid_plant_confidence}%" if plantid_plant_name else "N/A",
            "api_used": "Plant.id API v3",
            "status": "success",
            **treatment
        }

        if not plantid_response:
            response["warning"] = "Plant.id API unavailable - using model prediction"
        elif source == "your_model_fallback":
            response["warning"] = f"Neither prediction met confidence threshold ({CONFIDENCE_THRESHOLD}%) - using model prediction"

        return jsonify(response), 200

    except Exception as e:
        return jsonify({
            "error": str(e),
            "message": "An error occurred during processing",
            "status": "error",
            "timestamp": datetime.now().isoformat()
        }), 500

    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

@app.route("/", methods=["GET"])
def health_check():
    return jsonify({
        "status": "active",
        "service": "Plant Disease Detection API",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": bool(model),
        "plantid_api_configured": bool(PLANT_ID_API_KEY),
        "confidence_threshold": f"{CONFIDENCE_THRESHOLD}%"
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
