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

# Add path to model
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'modeltrainer/outputModel')

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
CORS(app, resources={r"/api/*": {"origins": "*"}})  # Allow all origins for /api

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize models
def init_models():
    """Initialize and load all required models."""
    try:
        # Load pretrained models
        bert_tokenizer = AutoTokenizer.from_pretrained("tbs17/MathBERT")
        bert_model = AutoModel.from_pretrained("tbs17/MathBERT").to(device)
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Initialize our model
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
        
        # Load trained weights
        checkpoint_path = os.path.join(MODEL_PATH, "multimodal_model_final.pth")
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()  # Set to evaluation mode
            logging.info("Modelo multimodal cargado exitosamente")
        else:
            raise FileNotFoundError(f"No se encontró el checkpoint en {checkpoint_path}")
            
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

# Set OpenAI API key from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set")


def get_prompt_content():
    """Read prompt content from an external text file."""
    try:
        with open('prompt.txt', 'r') as file:
            return file.read().strip()
    except Exception as e:
        logging.error(f"Error reading prompt file: {e}")
        return "ERROR: Unable to load prompt content."


def execute_matplotlib_code(code):
    """Execute matplotlib code and return the plot as base64 image."""
    try:
        # Remove any plt.show() calls from the code
        code = re.sub(r'plt\.show\(\s*\)', '', code)
        
        # Create a new figure
        plt.close('all')
        
        # Execute the modified code
        exec(code, globals())
        
        # Save plot to buffer
        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        
        # Encode to base64
        graphic = base64.b64encode(image_png).decode('utf-8')
        
        return graphic
    except Exception as e:
        logging.error(f"Error executing matplotlib code: {e}")
        raise

def process_prompt_with_chatgpt(prompt):
    """Process the user's prompt with GPT and return the response."""
    try:
        prompt_content = get_prompt_content()
        if "ERROR" in prompt_content:
            return {"type": "error", "content": prompt_content}

        client = openai.OpenAI()
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": prompt_content},
                {"role": "user", "content": prompt},
            ],
            max_tokens=1000,
            temperature=0.7,
        )
        raw_response = response.choices[0].message.content.strip()

        # Log GPT raw response for debugging
        logging.debug(f"GPT raw response: {raw_response}")

        # First check for Python code blocks
        code_blocks = re.findall(r"```(?:python)?\s*(.*?)```", raw_response, re.DOTALL)
        
        if code_blocks:
            # Check if any code block contains matplotlib
            for code in code_blocks:
                if 'matplotlib' in code or 'plt.' in code:
                    try:
                        # Execute matplotlib code and get image
                        image_base64 = execute_matplotlib_code(code)
                        return {
                            "type": "plot",
                            "content": code.strip(),
                            "image": image_base64
                        }
                    except Exception as e:
                        logging.error(f"Failed to execute matplotlib code: {e}")
                        return {"type": "error", "content": str(e)}
            
            # If no matplotlib found, return first code block as regular code
            return {"type": "code", "content": code_blocks[0].strip()}
        else:
            # No code blocks found, return as text
            return {"type": "text", "content": raw_response}
    except Exception as e:
        logging.error(f"Error with OpenAI API: {e}")
        return {"type": "error", "content": str(e)}


@app.route('/api/generate', methods=['POST'])
def generate():
    data = request.json
    logging.debug(f"Received request payload: {data}")
    prompt = data.get('prompt', '').strip()

    if not prompt:
        logging.error("No prompt provided in the request.")
        return jsonify({'error': 'No prompt provided.'}), 400

    logging.info(f"Received prompt: {prompt}")

    # Process the prompt with GPT
    gpt_response = process_prompt_with_chatgpt(prompt)

    if gpt_response["type"] == "error":
        logging.error(f"Error in GPT response: {gpt_response['content']}")
        return jsonify({'error': gpt_response['content']}), 500

    if gpt_response["type"] == "text":
        return jsonify({'type': 'text', 'content': gpt_response['content']})

    if gpt_response["type"] == "plot":
        return jsonify({
            'type': 'plot',
            'content': gpt_response['content'],
            'image': gpt_response['image']
        })
    elif gpt_response["type"] == "code":
        return jsonify({'type': 'code', 'content': gpt_response['content']})

    return jsonify({'error': 'Unexpected response type from GPT.'}), 500


def process_with_multimodal_model(text, image):
    """Process text and image with the multimodal model."""
    try:
        if models is None:
            return {"type": "error", "content": "Models not initialized properly"}
            
        # Prepare text input
        text_inputs = models['bert_tokenizer'](
            text,
            padding='max_length',
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(device)
        
        # Prepare image input
        image_inputs = models['clip_processor'](images=image, return_tensors="pt").to(device)
        
        with torch.no_grad():
            # Get embeddings
            text_outputs = models['bert_model'](**text_inputs).pooler_output
            image_outputs = models['clip_model'].get_image_features(**image_inputs)
            
            # Get model prediction
            outputs = models['multimodal_model'](text_outputs, image_outputs)
            probabilities = torch.softmax(outputs, dim=1)
            prediction = torch.argmax(probabilities, dim=1)
            confidence = probabilities[0][prediction[0]].item()
            
            return {
                "type": "prediction",
                "content": {
                    "prediction": prediction[0].item(),
                    "confidence": confidence
                }
            }
    except Exception as e:
        logging.error(f"Error in multimodal processing: {e}")
        return {"type": "error", "content": str(e)}

@app.route('/api/predict', methods=['POST'])
def predict():
    """Endpoint for multimodal predictions."""
    try:
        # Get text and image from request
        data = request.json
        text = data.get('text', '').strip()
        image_data = data.get('image', '')  # Expecting base64 encoded image
        
        if not text or not image_data:
            return jsonify({'error': 'Both text and image are required'}), 400
            
        # Decode base64 image
        try:
            image_bytes = base64.b64decode(image_data.split(',')[1])
            image = Image.open(BytesIO(image_bytes)).convert('RGB')
        except Exception as e:
            logging.error(f"Error decoding image: {e}")
            return jsonify({'error': 'Invalid image data'}), 400
            
        # Process with model
        result = process_with_multimodal_model(text, image)
        
        if result["type"] == "error":
            return jsonify({'error': result['content']}), 500
            
        return jsonify(result)
        
    except Exception as e:
        logging.error(f"Error in predict endpoint: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.getenv("PORT", 2604))
    logging.info(f"Starting Flask app on port {port}")
    app.run(debug=True, port=port)
