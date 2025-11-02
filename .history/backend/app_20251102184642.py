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
        api_key = os.getenv('DEEPSEEK_API_KEY', '').strip()
        if not api_key:
            return jsonify({'error': 'DeepSeek API key missing. Set DEEPSEEK_API_KEY in backend/.env and restart the server.'}), 500

        base_url = os.getenv('DEEPSEEK_BASE_URL', 'https://api.deepseek.com').rstrip('/')
        model_name = os.getenv('DEEPSEEK_MODEL', 'deepseek-chat').strip()

        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'Authorization': f"Bearer {api_key}",
            'X-API-Key': api_key  # some providers accept this header instead of Authorization
        }
        body = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": prompt_content},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.9,
            "max_tokens": 1000
        }
        # try primary path, then fallback to OpenAI-style /v1 path if needed
        url_primary = f"{base_url}/chat/completions"
        url_v1 = f"{base_url}/v1/chat/completions"

        response = requests.post(url_primary, headers=headers, json=body, timeout=60)
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as http_err:
            status = response.status_code if response is not None else 'N/A'
            # fallback to /v1 path if 404/400 due to path or versioning
            if status in (400, 404):
                try:
                    response2 = requests.post(url_v1, headers=headers, json=body, timeout=60)
                    response2.raise_for_status()
                    data = response2.json()
                    raw_response = data['choices'][0]['message']['content'].strip()
                    code_blocks = re.findall(r"```(?:python)?\s*(.*?)```", raw_response, re.DOTALL)
                    if code_blocks:
                        for code in code_blocks:
                            if 'matplotlib' in code or 'plt.' in code:
                                image_base64 = execute_matplotlib_code(code)
                                return jsonify({"type": "plot", "content": code.strip(), "image": image_base64})
                        return jsonify({"type": "code", "content": code_blocks[0].strip()})
                    return jsonify({"type": "text", "content": raw_response})
                except Exception:
                    pass
            if status == 401:
                return jsonify({'error': 'DeepSeek 401 Unauthorized. Verify DEEPSEEK_API_KEY, model access, and account status.'}), 502
            return jsonify({'error': f'DeepSeek API error ({status}): {response.text[:600] if response is not None else str(http_err)}'}), 502
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


@app.route('/api/health', methods=['GET'])
@log_activity('health')
def health():
    """Simple health check with version and device info"""
    try:
        return jsonify({
            'status': 'ok',
            'version': get_version(),
            'cuda': torch.cuda.is_available(),
            'device': str(device),
            'deepseek': {
                'configured': bool(os.getenv('DEEPSEEK_API_KEY')),
                'base_url': os.getenv('DEEPSEEK_BASE_URL', 'https://api.deepseek.com'),
                'model': os.getenv('DEEPSEEK_MODEL', 'deepseek-chat')
            }
        })
    except Exception as e:
        logging.error(f"Error in /api/health: {e}")
        return jsonify({'error': str(e)}), 500


# ARANGO

from arango import ArangoClient

# Configuración de conexión
ARANGO_URL = os.getenv("ARANGO_URL", "http://localhost:8529")
ARANGO_DB = os.getenv("ARANGO_DB", "somap")
ARANGO_USER = os.getenv("ARANGO_USER", "root")
ARANGO_PASS = os.getenv("ARANGO_PASS", "asdBGT788")

client = ArangoClient(hosts=ARANGO_URL)
db = client.db(ARANGO_DB, username=ARANGO_USER, password=ARANGO_PASS)

@app.route('/api/somap/<collection>', methods=['GET'])
def get_collection(collection):
    try:
        col = db.collection(collection)
        docs = [doc for doc in col.all()]
        columns = list(docs[0].keys()) if docs else []
        return jsonify({"rows": docs, "columns": columns})
    except Exception as e:
        logging.error(f"Error fetching collection {collection}: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/somap/<collection>', methods=['POST'])
def save_collection(collection):
    try:
        data = request.get_json()
        rows = data.get("rows", [])
        col = db.collection(collection)
        # Merge/update: upsert por _key si existe, inserta si no
        for row in rows:
            if '_key' in row:
                if col.has(row['_key']):
                    col.update(row)
                else:
                    col.insert(row)
            else:
                col.insert(row)
        return jsonify({"success": True})
    except Exception as e:
        logging.error(f"Error saving collection {collection}: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/somap/<collection>/<key>', methods=['DELETE'])
def delete_document(collection, key):
    try:
        col = db.collection(collection)
        col.delete(key)
        return jsonify({"success": True})
    except Exception as e:
        logging.error(f"Error deleting from {collection}: {e}")
        return jsonify({"error": str(e)}), 500

        # END ARANGO


@app.route('/api/deepseek/verify', methods=['GET'])
@log_activity('deepseek-verify')
def deepseek_verify():
    """Verify DeepSeek credentials by listing models.
    Does not expose sensitive data; returns status and minimal info.
    """
    try:
        api_key = os.getenv('DEEPSEEK_API_KEY', '').strip()
        if not api_key:
            return jsonify({'ok': False, 'error': 'Missing DEEPSEEK_API_KEY'}), 500

        base_url = os.getenv('DEEPSEEK_BASE_URL', 'https://api.deepseek.com').rstrip('/')
        headers = {
            'Accept': 'application/json',
            'Authorization': f'Bearer {api_key}',
            'X-API-Key': api_key
        }
        # Try /v1 first (OpenAI style), then fallback to legacy
        for path in ('/v1/models', '/models'):
            url = f"{base_url}{path}"
            r = requests.get(url, headers=headers, timeout=30)
            if r.status_code == 200:
                payload = r.json()
                model_count = len(payload.get('data', [])) if isinstance(payload, dict) and 'data' in payload else 'n/a'
                return jsonify({'ok': True, 'endpoint': url, 'models': model_count})
            elif r.status_code == 401:
                return jsonify({'ok': False, 'status': 401, 'error': 'Unauthorized'}), 502
        return jsonify({'ok': False, 'status': r.status_code, 'error': r.text[:300]}), 502
    except Exception as e:
        logging.error(f"Error in /api/deepseek/verify: {e}")
        return jsonify({'ok': False, 'error': str(e)}), 500


if __name__ == '__main__':
    port = int(os.getenv("PORT", 10000))
    logging.info(f"Starting Flask app on port {port}")
    app.run(host='0.0.0.0', port=port, debug=True)