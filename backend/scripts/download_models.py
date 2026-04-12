import os
import torch
import logging
from diffusers import AutoPipelineForImage2Image, StableAudioPipeline
from dotenv import load_dotenv

# Set cache directory to the project's local .cache folder
HF_CACHE_DIR = os.path.join(os.getcwd(), ".cache", "huggingface")
os.makedirs(HF_CACHE_DIR, exist_ok=True)
os.environ["HF_HOME"] = HF_CACHE_DIR

# Suppress some verbose output
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("download_models")

# Load environment variables from .env
# Assuming we are running from the backend directory or the root
load_dotenv()
if not os.path.exists('.env'):
    # Try looking one level up if running from scripts/
    load_dotenv('../.env')

def download():
    logger.info("--- SOOG Model Pre-download Script ---")
    
    # 1. Sketch Model (Stable Diffusion)
    # Get config from env or defaults
    sketch_model = os.getenv('SOOG_SKETCH_MODEL', 'OFA-Sys/small-stable-diffusion-v0')
    logger.info(f"Checking Sketch Model: {sketch_model}...")
    try:
        AutoPipelineForImage2Image.from_pretrained(
            sketch_model, 
            cache_dir=HF_CACHE_DIR,
            trust_remote_code=True
        )
        logger.info("✓ Sketch model ready.")
    except Exception as e:
        logger.error(f"✕ Failed to download Sketch Model: {e}")

    # 2. Stable Audio Open
    sound_model = os.getenv('SOOG_SOUND_MODEL', 'stabilityai/stable-audio-open-1.0')
    token = os.getenv('HF_TOKEN') or os.getenv('STABLE_AUDIO_OPEN_TOKEN')
    
    logger.info(f"Checking Sound Model: {sound_model}...")
    if not token:
        logger.warning("! Warning: No HF_TOKEN or STABLE_AUDIO_OPEN_TOKEN found in .env.")
        logger.warning("  Stable Audio Open is a gated model and requires a token for first download.")
    
    try:
        StableAudioPipeline.from_pretrained(
            sound_model,
            token=token,
            cache_dir=HF_CACHE_DIR,
            trust_remote_code=True
        )
        logger.info("✓ Sound model ready.")
    except Exception as e:
        logger.error(f"✕ Failed to download Sound Model: {e}")
        if "401" in str(e) or "403" in str(e):
            logger.error("  This looks like an authentication error. Please check your token and make sure you have accepted the model terms on Hugging Face.")

    logger.info("\nModels are cached in: " + HF_CACHE_DIR)

if __name__ == "__main__":
    download()
