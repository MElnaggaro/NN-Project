from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io
import base64

app = Flask(__name__)
CORS(app)

# Emotion labels (MUST match training order)
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# ===================== LOAD MODEL =====================
print("=" * 60)
print("LOADING EMOTION RECOGNITION MODEL")
print("=" * 60)

model = load_model("emotion_model_complete.h5", compile=False)

print("✓ Model loaded successfully")
print(f"Input shape : {model.input_shape}")
print(f"Output shape: {model.output_shape}")

print("=" * 60)
print("MODEL READY")
print("=" * 60 + "\n")


# ===================== PREPROCESS =====================
def preprocess_image(image_data):
    """
    Preprocess image for model input
    NOTE: Rescaling is already inside the model
    """
    # Decode base64
    img_data = base64.b64decode(image_data.split(',')[1])
    img = Image.open(io.BytesIO(img_data))

    # Convert to grayscale
    img = img.convert('L')

    # Resize
    img = img.resize((48, 48))

    # To numpy
    img_array = np.array(img, dtype=np.float32)

    # Shape: (1, 48, 48, 1)
    img_array = img_array.reshape(1, 48, 48, 1)

    return img_array


# ===================== ROUTES =====================
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        image_data = data.get('image')

        if not image_data:
            return jsonify({'error': 'No image provided'}), 400

        processed_image = preprocess_image(image_data)

        predictions = model.predict(processed_image, verbose=0)

        emotion_index = int(np.argmax(predictions[0]))
        emotion = EMOTIONS[emotion_index]
        confidence = float(predictions[0][emotion_index])

        all_predictions = {
            EMOTIONS[i]: float(predictions[0][i])
            for i in range(len(EMOTIONS))
        }

        return jsonify({
            'emotion': emotion,
            'confidence': confidence,
            'all_predictions': all_predictions
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': True,
        'emotions': EMOTIONS
    })


# ===================== RUN SERVER =====================
if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("🚀 FLASK SERVER RUNNING")
    print("=" * 60)
    print("Server : http://localhost:5000")
    print("Health : http://localhost:5000/health")
    print("=" * 60 + "\n")

    app.run(debug=True, port=5000, host='0.0.0.0')
