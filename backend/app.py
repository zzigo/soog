from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from dotenv import load_dotenv
from logger import log_activity, get_logs, format_logs_html
import requests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
import base64
import os
import logging
import re
import sys
import torch
import scipy
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer, AutoModel
from torch import nn

HF_CACHE_DIR = os.path.join(os.getcwd(), ".cache", "huggingface")
OFFLOAD_DIR = os.path.join(os.getcwd(), "offload")
os.makedirs(HF_CACHE_DIR, exist_ok=True)
os.makedirs(OFFLOAD_DIR, exist_ok=True)
os.environ["HF_HOME"] = HF_CACHE_DIR

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(os.path.dirname(__file__), 'api.log'))
    ]
)

load_dotenv()

app = Flask(__name__)
CORS(app, resources={
    r"/api/*": {"origins": "*"},
    r"/log": {"origins": "*"}
})

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_version():
    try:
        with open(os.path.join(os.path.dirname(__file__), 'version.txt'), 'r') as file:
            return file.read().strip()
    except Exception as e:
        logging.error(f"Error reading version file: {e}")
        return "0.0.0"

def get_prompt_content():
    try:
        with open(os.path.join(os.path.dirname(__file__), 'prompt.txt'), 'r') as file:
            return file.read().strip()
    except Exception as e:
        logging.error(f"Error reading prompt file: {e}")
        return "ERROR: Unable to load prompt content."

def execute_matplotlib_code(code):
    try:
        code = re.sub(r'plt\.show\(\s*\)', '', code)
        plt.close('all')
        exec(code, globals())
        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode('utf-8')
    except Exception as e:
        logging.error(f"Error executing matplotlib code: {e}")
        raise

def init_models():
    try:
        bert_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased", cache_dir=HF_CACHE_DIR)
        bert_model = AutoModel.from_pretrained("distilbert-base-uncased", cache_dir=HF_CACHE_DIR).to(device)
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16", cache_dir=HF_CACHE_DIR).to(device)
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16", cache_dir=HF_CACHE_DIR)

        class MultimodalAttentionModel(nn.Module):
            def __init__(self, text_hidden_size=768, image_hidden_size=512, combined_hidden_size=256):
                super().__init__()
                self.text_fc = nn.Linear(text_hidden_size, combined_hidden_size)
                self.image_fc = nn.Linear(image_hidden_size, combined_hidden_size)
                self.attention = nn.MultiheadAttention(embed_dim=combined_hidden_size, num_heads=4)
                self.classifier = nn.Linear(combined_hidden_size, 10)

            def forward(self, text_features, image_features):
                text_out = self.text_fc(text_features)
                image_out = self.image_fc(image_features)
                attention_out, _ = self.attention(text_out.unsqueeze(1), image_out.unsqueeze(1), image_out.unsqueeze(1))
                return self.classifier(attention_out.squeeze(1))

        multimodal_model = MultimodalAttentionModel().to(device)
        ckpt_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "modeltrainer/outputModel/multimodal_model_final.pth")
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location=device)
            multimodal_model.load_state_dict(checkpoint['model_state_dict'])
            multimodal_model.eval()
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

models = {}

def load_models_if_needed():
    global models
    if not models:
        models.update(init_models())

def cleanup_models():
    global models
    if models:
        for name, model in models.items():
            if hasattr(model, 'to'):
                model.to('cpu')
                del model
        models.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc
        gc.collect()

@app.route('/log', methods=['GET'])
def view_logs():
    try:
        logs = get_logs(limit=100)
        return Response(format_logs_html(logs), content_type='text/html')
    except Exception as e:
        logging.error(f"Error viewing logs: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/generate', methods=['POST'])
@log_activity('generate')
def generate():
    data = request.json
    prompt = data.get('prompt', '').strip()
    if not prompt:
        return jsonify({'error': 'No prompt provided.'}), 400
    try:
        prompt_content = get_prompt_content()
        if "ERROR" in prompt_content:
            return jsonify({'error': prompt_content}), 500
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f"Bearer {os.getenv('DEEPSEEK_API_KEY')}"
        }
        body = {
            "model": "deepseek-chat",  # o el modelo exacto que uses
            "messages": [
                {"role": "system", "content": prompt_content},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.9,
            "max_tokens": 1000
        }
        response = requests.post("https://api.deepseek.com/chat/completions", headers=headers, json=body)
        response.raise_for_status()
        data = response.json()
        raw_response = data['choices'][0]['message']['content'].strip()
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
@log_activity('predict')
def predict():
    data = request.json
    text = data.get('text', '').strip()
    image_data = data.get('image', '')
    if not text or not image_data:
        return jsonify({'error': 'Both text and image are required.'}), 400
    try:
        load_models_if_needed()
        image_bytes = base64.b64decode(image_data.split(',')[1])
        image = Image.open(BytesIO(image_bytes)).convert('RGB')
        text_inputs = models['bert_tokenizer'](text, return_tensors="pt", padding=True, truncation=True)
        image_inputs = models['clip_processor'](images=image, return_tensors="pt")
        with torch.no_grad():
            text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
            image_inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in image_inputs.items()}
            text_features = models['bert_model'](**text_inputs).pooler_output
            image_features = models['clip_model'].get_image_features(**image_inputs)
            outputs = models['multimodal_model'](text_features, image_features)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        cleanup_models()
        return jsonify({'prediction': predicted_class, 'confidence': confidence})
    except Exception as e:
        logging.error(f"Error in /api/predict: {e}")
        cleanup_models()
        return jsonify({'error': str(e)}), 500

@app.route('/api/version', methods=['GET'])
@log_activity('version')
def version():
    try:
        return jsonify({'version': get_version()})
    except Exception as e:
        logging.error(f"Error in /api/version: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.getenv("PORT", 10000))
    logging.info(f"Starting Flask app on port {port}")
    app.run(host='0.0.0.0', port=port, debug=True)