import os
import numpy as np
import pickle
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from tensorflow.keras.models import load_model
from scipy.spatial.distance import pdist

# Initialize Flask, pointing to the 'templates' folder for HTML
app = Flask(__name__, template_folder='templates')
CORS(app)

# --- 1. Load Resources ---
MODEL_PATH = 'relational_model.h5'
LABELS_PATH = 'landmark_labels.pkl'

print("Loading model...")
try:
    model = load_model(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

print("Loading labels...")
try:
    with open(LABELS_PATH, 'rb') as f:
        label_encoder = pickle.load(f)
    print("Labels loaded successfully.")
except Exception as e:
    print(f"Error loading labels: {e}")
    label_encoder = None

# --- 2. Feature Extraction Logic ---
def extract_relational_features(landmarks):
    """
    Computes pairwise Euclidean distances between landmarks.
    Expects landmarks to be a list of [x, y] coordinates.
    """
    landmarks = np.array(landmarks)
    distances = pdist(landmarks, metric='euclidean')
    return distances

# --- 3. Routes ---

@app.route('/')
def home():
    """Serves the Frontend Website"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handles the AI Prediction"""
    if not model:
        return jsonify({'error': 'Model not loaded'}), 500

    data = request.json
    if 'landmarks' not in data:
        return jsonify({'error': 'No landmarks provided'}), 400

    try:
        raw_landmarks = data['landmarks']
        points = [[l['x'], l['y']] for l in raw_landmarks]

        features = extract_relational_features(points)
        features_for_model = np.expand_dims(features, axis=0)

        prediction_probs = model.predict(features_for_model)
        prediction_idx = np.argmax(prediction_probs)
        confidence = float(np.max(prediction_probs))

        result_label = "Unknown"
        if label_encoder:
            result_label = label_encoder.inverse_transform([prediction_idx])[0]
        else:
            result_label = str(prediction_idx)

        if confidence < 0.6:
            result_label = "Unsure"

        return jsonify({
            'sign': result_label,
            'confidence': confidence
        })

    except Exception as e:
        print(f"Prediction Error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)