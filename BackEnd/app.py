from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from PIL import Image
import io
import base64
import h5py

app = Flask(__name__)
CORS(app)

# Emotion labels
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

def create_model():
    """Create model architecture"""
    model = keras.Sequential([
        layers.Rescaling(1./255, input_shape=(48, 48, 1)),
        
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        layers.GlobalAveragePooling2D(),
        
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        layers.Dense(7, activation='softmax')
    ])
    
    return model

def load_weights_from_h5_group(model, h5_file):
    """
    Manually load weights from H5 file with custom structure
    """
    print("\nAttempting manual weight loading...")
    
    with h5py.File(h5_file, 'r') as f:
        if 'model_weights' not in f:
            raise ValueError("'model_weights' group not found in H5 file")
        
        weights_group = f['model_weights']
        
        # Get all layer names from the file
        layer_names = [key for key in weights_group.keys() 
                      if not key.startswith('top_level')]
        
        print(f"Found {len(layer_names)} layers in file")
        
        # Map of layers that have weights
        layers_with_weights = [l for l in model.layers if len(l.weights) > 0]
        print(f"Model has {len(layers_with_weights)} layers with weights")
        
        # Load weights for each layer
        loaded_count = 0
        for model_layer in layers_with_weights:
            layer_name = model_layer.name
            
            # Find matching layer in file
            if layer_name in weights_group:
                layer_group = weights_group[layer_name]
                
                # Navigate through the structure
                if 'sequential_1' in layer_group:
                    layer_group = layer_group['sequential_1']
                
                if layer_name in layer_group:
                    layer_group = layer_group[layer_name]
                    
                    # Load weights
                    weight_names = list(layer_group.keys())
                    weight_values = [np.array(layer_group[w]) for w in weight_names]
                    
                    if len(weight_values) == len(model_layer.weights):
                        model_layer.set_weights(weight_values)
                        loaded_count += 1
                        print(f"  ✓ Loaded {layer_name}: {len(weight_values)} tensors")
                    else:
                        print(f"  ✗ Skipped {layer_name}: weight count mismatch")
            else:
                print(f"  ⊘ {layer_name} not found in file")
        
        print(f"\nSuccessfully loaded weights for {loaded_count}/{len(layers_with_weights)} layers")
        
        if loaded_count < len(layers_with_weights) * 0.8:  # Less than 80%
            raise ValueError(f"Only loaded {loaded_count} out of {len(layers_with_weights)} layers")

# Initialize model
print("="*60)
print("INITIALIZING EMOTION RECOGNITION MODEL")
print("="*60)

print("\n1. Creating model architecture...")
model = create_model()

print("2. Building model...")
model.build((None, 48, 48, 1))

print("3. Loading weights from model.weights.h5...")
try:
    # Try standard loading first
    model.load_weights('model.weights.h5')
    print("   ✓ Standard loading succeeded!")
except Exception as e1:
    print(f"   ⊘ Standard loading failed: {str(e1)[:100]}...")
    
    try:
        # Try manual loading
        load_weights_from_h5_group(model, 'model.weights.h5')
        print("   ✓ Manual loading succeeded!")
    except Exception as e2:
        print(f"   ✗ Manual loading failed: {e2}")
        
        # Last resort - try loading with skip_mismatch
        try:
            print("\n   Trying with skip_mismatch=True...")
            model.load_weights('model.weights.h5', by_name=True, skip_mismatch=True)
            print("   ✓ Partial loading succeeded (some layers may be uninitialized)")
        except Exception as e3:
            print(f"   ✗ All loading methods failed")
            print("\nYou need to provide the EXACT training code used to create this model.")
            print("The architecture must match perfectly for weights to load.")
            raise

print("\n4. Model Summary:")
print(f"   - Input shape: {model.input_shape}")
print(f"   - Output shape: {model.output_shape}")
print(f"   - Total parameters: {model.count_params():,}")

print("\n" + "="*60)
print("MODEL READY!")
print("="*60 + "\n")

def preprocess_image(image_data):
    """Preprocess image for model input"""
    img_data = base64.b64decode(image_data.split(',')[1])
    img = Image.open(io.BytesIO(img_data))
    img = img.convert('L')
    img = img.resize((48, 48))
    img_array = np.array(img, dtype=np.float32)
    img_array = img_array.reshape(1, 48, 48, 1)
    return img_array

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
        
        print("\n" + "-"*50)
        print("Processing prediction request...")
        processed_image = preprocess_image(image_data)
        
        predictions = model.predict(processed_image, verbose=0)
        
        emotion_index = np.argmax(predictions[0])
        emotion = EMOTIONS[emotion_index]
        confidence = float(predictions[0][emotion_index])
        
        all_predictions = {
            EMOTIONS[i]: float(predictions[0][i]) 
            for i in range(len(EMOTIONS))
        }
        
        print(f"Prediction: {emotion} ({confidence:.1%})")
        for emo, prob in sorted(all_predictions.items(), key=lambda x: x[1], reverse=True)[:3]:
            print(f"  {emo}: {prob:.1%}")
        print("-"*50)
        
        return jsonify({
            'emotion': emotion,
            'confidence': confidence,
            'all_predictions': all_predictions
        })
        
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'emotions': EMOTIONS
    })

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 STARTING FLASK SERVER")
    print("="*60)
    print(f"Server: http://localhost:5000")
    print(f"Health: http://localhost:5000/health")
    print("="*60 + "\n")
    
    app.run(debug=True, port=5000, host='0.0.0.0')