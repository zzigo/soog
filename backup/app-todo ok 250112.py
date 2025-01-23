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


if __name__ == '__main__':
    port = int(os.getenv("PORT", 2604))
    logging.info(f"Starting Flask app on port {port}")
    app.run(debug=True, port=port)
