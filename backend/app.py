from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import openai
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for Matplotlib
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
import base64
import os
import logging
import re
import sys
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer, AutoModel
from torch import nn

# Set Hugging Face cache directory
HF_CACHE_DIR = os.path.join(os.getcwd(), ".cache", "huggingface")
os.environ["HF_HOME"] = HF_CACHE_DIR

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('app.log')
    ]
)

# Load environment variables from .env file
load_dotenv()

# Initialize Flask app and enable CORS
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize models
def init_models():
    """Initialize and load all required models."""
    try:
        # Load smaller pretrained models for efficiency
        bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", cache_dir=HF_CACHE_DIR)
        bert_model = AutoModel.from_pretrained("bert-base-uncased", cache_dir=HF_CACHE_DIR).to(device)
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16", cache_dir=HF_CACHE_DIR).to(device)
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16", cache_dir=HF_CACHE_DIR)
        
        # Initialize multimodal model
        class MultimodalAttentionModel(nn.Module):
            def __init__(self, text_hidden_size=768, image_hidden_size=512, combined_hidden_size=256):
                super(MultimodalAttentionModel, self).__init__()
                self.text_fc = nn.Linear(text_hidden_size, combined_hidden_size)
                self.image_fc = nn.Linear(image_hidden_size, combined_hidden_size)
                self.attention = nn.MultiheadAttention(embed_dim=combined_hidden_size, num_heads=4)
                self.classifier = nn.Linear(combined_hidden_size, 10)

            def forward(self, text_features, image_features):
                text_out = self.text_fc(text_features)
                image_out = self.image_fc(image_features)
                attention_out, _ = self.attention(text_out.unsqueeze(1), image_out.unsqueeze(1), image_out.unsqueeze(1))
                combined = attention_out.squeeze(1)
                return self.classifier(combined)

        # Create model instance
        model = MultimodalAttentionModel().to(device)
        
        # Load trained weights (if available)
        checkpoint_path = "./modeltrainer/outputModel/multimodal_model_final.pth"
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            logging.info("Multimodal model loaded successfully")
        else:
            logging.warning(f"No checkpoint found at {checkpoint_path}. Using untrained model.")
            
        return {
            'bert_tokenizer': bert_tokenizer,
            'bert_model': bert_model,
            'clip_model': clip_model,
            'clip_processor': clip_processor,
            'multimodal_model': model
        }
    except Exception as e:
        logging.error(f"Error initializing models: {e}")
        raise

# Initialize all models
try:
    models = init_models()
    logging.info("All models initialized successfully")
except Exception as e:
    logging.error(f"Failed to initialize models: {e}")
    models = None

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

# Endpoint for GPT interaction
@app.route('/api/generate', methods=['POST'])
def generate():
    data = request.json
    prompt = data.get('prompt', '').strip()
    if not prompt:
        return jsonify({'error': 'No prompt provided.'}), 400

    try:
        # Process with OpenAI GPT
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.7
        )
        return jsonify(response['choices'][0]['message']['content'])
    except Exception as e:
        logging.error(f"Error in OpenAI API: {e}")
        return jsonify({'error': str(e)}), 500

# Endpoint for multimodal predictions
@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get('text', '').strip()
    image_data = data.get('image', '')  # Base64 encoded image

    if not text or not image_data:
        return jsonify({'error': 'Both text and image are required.'}), 400

    try:
        # Decode and preprocess the image
        image_bytes = base64.b64decode(image_data.split(',')[1])
        image = Image.open(BytesIO(image_bytes)).convert('RGB')

        # Process with multimodal model
        text_inputs = models['bert_tokenizer'](text, return_tensors="pt", padding=True, truncation=True).to(device)
        image_inputs = models['clip_processor'](images=image, return_tensors="pt").to(device)

        with torch.no_grad():
            text_features = models['bert_model'](**text_inputs).pooler_output
            image_features = models['clip_model'].get_image_features(**image_inputs)
            outputs = models['multimodal_model'](text_features, image_features)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()

        return jsonify({
            'prediction': predicted_class,
            'confidence': confidence
        })
    except Exception as e:
        logging.error(f"Error in prediction: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.getenv("PORT", 2604))
    logging.info(f"Starting Flask app on port {port}")
    app.run(debug=True, port=port)