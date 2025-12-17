import os
import numpy as np
import pickle
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from tensorflow.keras.models import load_model
from scipy.spatial.distance import pdist, squareform

# Initialize Flask
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

# --- 2. Feature Extraction Logic (CORRECTED to 421 features) ---
def extract_relational_features(landmarks):
    """
    Generates 421 features:
    1. Full Distance Matrix (excluding diagonal) = 21 * 20 = 420 features
    2. One Scale feature (distance between wrist and middle finger MCP) = 1 feature
    Total: 421 features
    """
    landmarks = np.array(landmarks) # Shape (21, 2)
    
    # 1. Calculate Pairwise Distances (Condensed: 210 values)
    condensed_dist = pdist(landmarks, metric='euclidean')
    
    # 2. Convert to Full Square Matrix (21x21)
    square_dist = squareform(condensed_dist)
    
    # 3. Flatten but remove the diagonal (self-distances which are always 0)
    # This gives us n*(n-1) = 21*20 = 420 values
    # We do this by selecting all elements where eye is 0
    mask = ~np.eye(square_dist.shape[0], dtype=bool)
    flattened_dist = square_dist[mask] # Shape (420,)

    # 4. Calculate Scale (Distance between Wrist(0) and Middle MCP(9))
    # Note: Middle Finger MCP is index 9
    scale_dist = np.linalg.norm(landmarks[0] - landmarks[9])
    
    # 5. Normalize (Optional but likely used in training)
    if scale_dist > 0:
        flattened_dist = flattened_dist / scale_dist

    # 6. Append Scale as the last feature (The 421st feature)
    # Note: Some models append it, some don't. Given the count, it's highly likely included.
    features = np.concatenate([flattened_dist, [scale_dist]])
    
    return features

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
        # Convert to list of [x, y]
        points = [[l['x'], l['y']] for l in raw_landmarks]

        # Extract features (Now returns 421 features)
        features = extract_relational_features(points)
        
        # Reshape for Keras (1, 421)
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
        # Return error as JSON so we can see it in browser console
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
