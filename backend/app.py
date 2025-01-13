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

# Set Hugging Face cache and offload directories
HF_CACHE_DIR = os.path.join(os.getcwd(), ".cache", "huggingface")
OFFLOAD_DIR = os.path.join(os.getcwd(), "offload")
os.makedirs(HF_CACHE_DIR, exist_ok=True)
os.makedirs(OFFLOAD_DIR, exist_ok=True)
os.environ["HF_HOME"] = HF_CACHE_DIR

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('api.log')
    ]
)

# Load environment variables from .env file
load_dotenv()

# Initialize Flask app and enable CORS
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_version():
    """Read the current version from version.txt."""
    try:
        with open('version.txt', 'r') as file:
            return file.read().strip()
    except Exception as e:
        logging.error(f"Error reading version file: {e}")
        return "0.0.0"

def get_prompt_content():
    """Read the system prompt content from 'prompt.txt'."""
    try:
        with open('prompt.txt', 'r') as file:
            return file.read().strip()
    except Exception as e:
        logging.error(f"Error reading prompt file: {e}")
        return "ERROR: Unable to load prompt content."


def execute_matplotlib_code(code):
    """Execute matplotlib code and return the plot as a base64 image."""
    try:
        code = re.sub(r'plt\.show\(\s*\)', '', code)  # Remove plt.show() calls
        plt.close('all')
        exec(code, globals())
        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        return base64.b64encode(image_png).decode('utf-8')
    except Exception as e:
        logging.error(f"Error executing matplotlib code: {e}")
        raise


def init_models():
    """Initialize and load all required models."""
    try:
        # Load pretrained models
        bert_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased", cache_dir=HF_CACHE_DIR)
        bert_model = AutoModel.from_pretrained(
            "distilbert-base-uncased", 
            cache_dir=HF_CACHE_DIR, 
            device_map="auto", 
            offload_folder=OFFLOAD_DIR
        )
        clip_model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch16", 
            cache_dir=HF_CACHE_DIR, 
            device_map="auto", 
            offload_folder=OFFLOAD_DIR
        )
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

        # Load multimodal model
        multimodal_model = MultimodalAttentionModel().to(device)
        checkpoint_path = "./modeltrainer/outputModel/multimodal_model_final.pth"
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            multimodal_model.load_state_dict(checkpoint['model_state_dict'])
            multimodal_model.eval()
            logging.info("Multimodal model loaded successfully")
        else:
            logging.warning(f"No checkpoint found at {checkpoint_path}. Using untrained model.")

        return {
            'bert_tokenizer': bert_tokenizer,
            'bert_model': bert_model,
            'clip_model': clip_model,
            'clip_processor': clip_processor,
            'multimodal_model': multimodal_model
        }
    except Exception as e:
        logging.error(f"Error initializing models: {e}")
        raise


# Initialize models
try:
    models = init_models()
    logging.info("All models initialized successfully")
except Exception as e:
    logging.error(f"Failed to initialize models: {e}")
    models = None


@app.route('/api/generate', methods=['POST'])
def generate():
    """API endpoint to process prompts with GPT."""
    data = request.json
    prompt = data.get('prompt', '').strip()

    if not prompt:
        return jsonify({'error': 'No prompt provided.'}), 400

    try:
        prompt_content = get_prompt_content()
        if "ERROR" in prompt_content:
            return jsonify({'error': prompt_content}), 500

        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": prompt_content},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.9
        )
        raw_response = response['choices'][0]['message']['content'].strip()
        code_blocks = re.findall(r"```(?:python)?\s*(.*?)```", raw_response, re.DOTALL)

        if code_blocks:
            for code in code_blocks:
                if 'matplotlib' in code or 'plt.' in code:
                    image_base64 = execute_matplotlib_code(code)
                    return jsonify({"type": "plot", "content": code.strip(), "image": image_base64})
            return jsonify({"type": "code", "content": code_blocks[0].strip()})
        return jsonify({"type": "text", "content": raw_response})
    except Exception as e:
        logging.error(f"Error in /api/generate: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/predict', methods=['POST'])
def predict():
    """API endpoint for multimodal predictions."""
    data = request.json
    text = data.get('text', '').strip()
    image_data = data.get('image', '')

    if not text or not image_data:
        return jsonify({'error': 'Both text and image are required.'}), 400

    try:
        image_bytes = base64.b64decode(image_data.split(',')[1])
        image = Image.open(BytesIO(image_bytes)).convert('RGB')
        text_inputs = models['bert_tokenizer'](text, return_tensors="pt", padding=True, truncation=True).to(device)
        image_inputs = models['clip_processor'](images=image, return_tensors="pt").to(device)

        with torch.no_grad():
            text_features = models['bert_model'](**text_inputs).pooler_output
            image_features = models['clip_model'].get_image_features(**image_inputs)
            outputs = models['multimodal_model'](text_features, image_features)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()

        return jsonify({'prediction': predicted_class, 'confidence': confidence})
    except Exception as e:
        logging.error(f"Error in /api/predict: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/version', methods=['GET'])
def version():
    """API endpoint to get the current version."""
    try:
        current_version = get_version()
        return jsonify({'version': current_version})
    except Exception as e:
        logging.error(f"Error in /api/version: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.getenv("PORT", 10000))
    logging.info(f"Starting Flask app on port {port}")
    app.run(host='0.0.0.0', port=port, debug=True)
