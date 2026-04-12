from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from dotenv import load_dotenv
from logger import log_activity, get_logs, format_logs_html
import requests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Arc, Circle, Ellipse, FancyArrowPatch, Polygon, Rectangle, RegularPolygon
import numpy as np
from io import BytesIO
import base64
import os
import logging
import re
import sys
import torch
import scipy
import scipy.io.wavfile
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer, AutoModel
from torch import nn
from flask import send_from_directory
import json
import datetime
import importlib
import time
import hashlib
import threading

try:
    import trimesh  # for STL mode
except Exception:
    trimesh = None

try:
    from shapely.geometry import LineString as ShapelyLineString, Point as ShapelyPoint, Polygon as ShapelyPolygon
    from shapely.ops import unary_union as shapely_unary_union
except Exception:
    ShapelyLineString = None
    ShapelyPoint = None
    ShapelyPolygon = None
    shapely_unary_union = None

HF_CACHE_DIR = os.path.join(os.getcwd(), ".cache", "huggingface")
OFFLOAD_DIR = os.path.join(os.getcwd(), "offload")
os.makedirs(HF_CACHE_DIR, exist_ok=True)
os.makedirs(OFFLOAD_DIR, exist_ok=True)
os.environ["HF_HOME"] = HF_CACHE_DIR

# Gallery directory
GALLERY_DIR = os.path.join(OFFLOAD_DIR, 'gallery')
os.makedirs(GALLERY_DIR, exist_ok=True)
GALLERY_FILE_SUFFIXES = ('.json', '.png', '.txt', '.stl', '.sketch.png', '.wav', '.ogg')

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(os.path.dirname(__file__), 'api.log'))
    ]
)

load_dotenv()

# Default model for all generations.
DEFAULT_OLLAMA_MODEL = 'deepseek-r1:8b'
# Fallback model if the default one fails repeatedly.
FALLBACK_OLLAMA_MODEL = 'qwen2.5:7b-instruct'

OLLAMA_MODEL_OVERRIDE = None
SOOPUB_CONTEXT_CACHE = {}
SOOPUB_DOCS_CACHE = None
SOOPUB_GRAPH_CACHE = {}
SKETCH_PIPELINE = None
SKETCH_PIPELINE_MODEL = None
SOUND_PIPELINE = None
SOUND_PIPELINE_MODEL = None
GENERATION_PROGRESS = {}
GENERATION_PROGRESS_LOCK = threading.Lock()
SOOPUB_SECTION_LABELS = {
    '1mat': 'materials',
    '2obj': 'objects',
    '3agn': 'agents',
    '4int': 'interfaces',
    '5pri': 'principles of functions',
    '6env': 'environments',
    '7dict': 'dictionary'
}

app = Flask(__name__)
app.url_map.strict_slashes = False
CORS(app, resources={
    r"/api/*": {"origins": "*"},
    r"/log": {"origins": "*"}
})

device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

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


def get_specialized_prompt(name: str):
    """
    Load specialized prompt files for different generation stages.
    Available: materials, digitalFab, inferred-image
    """
    filename = f"prompt_{name}.txt"
    try:
        path = os.path.join(os.path.dirname(__file__), filename)
        if not os.path.isfile(path):
            logging.warning(f"Specialized prompt {filename} not found, using generic prompt.txt")
            return get_prompt_content()
        with open(path, 'r') as file:
            return file.read().strip()
    except Exception as e:
        logging.error(f"Error reading specialized prompt {filename}: {e}")
        return get_prompt_content()


def get_ollama_config():
    base_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434').strip().rstrip('/')
    if not base_url:
        base_url = 'http://localhost:11434'

    model_name = (OLLAMA_MODEL_OVERRIDE or os.getenv('OLLAMA_MODEL', DEFAULT_OLLAMA_MODEL)).strip()
    if not model_name:
        model_name = DEFAULT_OLLAMA_MODEL

    api_key = os.getenv('OLLAMA_API_KEY', '').strip()
    return base_url, model_name, api_key


def get_ollama_headers():
    _, _, api_key = get_ollama_config()
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }
    # Optional for reverse proxies that protect Ollama with bearer auth.
    if api_key:
        headers['Authorization'] = f'Bearer {api_key}'
        headers['X-API-Key'] = api_key
    return headers


def get_progress_ttl_seconds():
    return _env_int('SOOG_PROGRESS_TTL_SEC', 900, min_value=60, max_value=86400)


def _trim_generation_progress_locked(now_ts=None):
    now_ts = now_ts or time.time()
    ttl = get_progress_ttl_seconds()
    stale = [
        key for key, value in GENERATION_PROGRESS.items()
        if now_ts - float(value.get('updated_at') or 0) > ttl
    ]
    for key in stale:
        GENERATION_PROGRESS.pop(key, None)


def _short_reasoning_preview(text: str, max_chars: int = 280) -> str:
    normalized = re.sub(r'\s+', ' ', (text or '')).strip()
    if len(normalized) <= max_chars:
        return normalized
    return normalized[-max_chars:].lstrip()


def update_generation_progress(request_id: str, **fields):
    if not request_id:
        return
    now_ts = time.time()
    with GENERATION_PROGRESS_LOCK:
        _trim_generation_progress_locked(now_ts)
        current = dict(GENERATION_PROGRESS.get(request_id) or {})
        current.update(fields)
        current['updated_at'] = now_ts
        GENERATION_PROGRESS[request_id] = current


def get_generation_progress(request_id: str):
    if not request_id:
        return None
    now_ts = time.time()
    with GENERATION_PROGRESS_LOCK:
        _trim_generation_progress_locked(now_ts)
        current = GENERATION_PROGRESS.get(request_id)
        if not current:
            return None
        return dict(current)


def _env_int(name: str, default: int, min_value: int = None, max_value: int = None) -> int:
    raw = os.getenv(name, '').strip()
    if not raw:
        value = int(default)
    else:
        try:
            value = int(raw)
        except ValueError:
            value = int(default)
    if min_value is not None:
        value = max(min_value, value)
    if max_value is not None:
        value = min(max_value, value)
    return value


def _env_float(name: str, default: float, min_value: float = None, max_value: float = None) -> float:
    raw = os.getenv(name, '').strip()
    if not raw:
        value = float(default)
    else:
        try:
            value = float(raw)
        except ValueError:
            value = float(default)
    if min_value is not None:
        value = max(min_value, value)
    if max_value is not None:
        value = min(max_value, value)
    return value


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name, '').strip().lower()
    if not raw:
        return bool(default)
    return raw in ('1', 'true', 'yes', 'on')


def get_generation_limits():
    return {
        # LLM HTTP request timeout per call. Set to 0 to disable.
        'llm_timeout_sec': _env_int('OLLAMA_REQUEST_TIMEOUT_SEC', 90, min_value=0, max_value=86400),
        # LLM max generated tokens per call (controls latency without wall-clock timeout).
        'llm_max_tokens': _env_int('SOOG_LLM_MAX_TOKENS', 1000, min_value=64, max_value=4096),
        # Internal correction loops inside one generation.
        'max_attempts': _env_int('SOOG_LLM_MAX_ATTEMPTS', 2, min_value=1, max_value=6),
        # Absolute wall-clock budget for /api/generate.
        # Set to 0 to disable deadline (stress-testing mode).
        'deadline_sec': _env_int('SOOG_GENERATE_DEADLINE_SEC', 240, min_value=0, max_value=86400),
        # Optional second pass without soopub context (disabled by default to avoid long waits).
        'retry_without_context': _env_bool('SOOG_RETRY_WITHOUT_CONTEXT', default=False),
        # Max PNG bytes to include inline in /api/generate JSON (0 disables inline image payload).
        'max_inline_image_bytes': _env_int('SOOG_MAX_INLINE_IMAGE_BYTES', 600000, min_value=0, max_value=20000000)
    }


def ollama_thinking_enabled():
    return _env_bool('SOOG_OLLAMA_THINKING', default=True)


def get_sketch_config():
    width = _env_int('SOOG_SKETCH_WIDTH', 384, min_value=128, max_value=1024)
    height = _env_int('SOOG_SKETCH_HEIGHT', 384, min_value=128, max_value=1024)
    width = max(128, (width // 8) * 8)
    height = max(128, (height // 8) * 8)
    return {
        'enabled': _env_bool('SOOG_SKETCH_ENABLED', default=True),
        'model_id': (os.getenv('SOOG_SKETCH_MODEL', 'OFA-Sys/small-stable-diffusion-v0') or '').strip() or 'OFA-Sys/small-stable-diffusion-v0',
        'width': width,
        'height': height,
        'steps': _env_int('SOOG_SKETCH_STEPS', 8, min_value=4, max_value=60),
        'strength': _env_float('SOOG_SKETCH_STRENGTH', 0.85, min_value=0.2, max_value=0.98),
        'guidance_scale': _env_float('SOOG_SKETCH_GUIDANCE_SCALE', 8.5, min_value=1.0, max_value=20.0),
        'cache_pipeline': _env_bool('SOOG_SKETCH_CACHE_PIPELINE', default=True)
    }


def _read_context_excerpt(path: str, max_chars: int = 3000):
    if not os.path.isfile(path):
        return ""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
    except Exception:
        return ""
    if not content:
        return ""
    return content[:max_chars]


def _keyword_tokens(text: str):
    return set(re.findall(r"[a-zA-Z]{4,}", (text or "").lower()))


def _load_soopub_docs():
    global SOOPUB_DOCS_CACHE
    if SOOPUB_DOCS_CACHE is not None:
        return SOOPUB_DOCS_CACHE

    base = os.path.join(os.path.dirname(__file__), 'soopub')
    if not os.path.isdir(base):
        SOOPUB_DOCS_CACHE = []
        return SOOPUB_DOCS_CACHE

    docs = []
    for root, _, files in os.walk(base):
        for filename in files:
            if not filename.lower().endswith(('.md', '.txt')):
                continue
            fp = os.path.join(root, filename)
            excerpt = _read_context_excerpt(fp, max_chars=2200)
            if not excerpt:
                continue
            label = os.path.relpath(fp, os.path.dirname(__file__))
            docs.append({
                'label': label,
                'text': excerpt,
                'tokens': _keyword_tokens(excerpt)
            })
    SOOPUB_DOCS_CACHE = docs
    return SOOPUB_DOCS_CACHE


def get_soopub_context(query_text: str = "", max_docs: int = 4, max_chars: int = 5600):
    """
    Load compact contextual notes from soopub for stronger organological guidance.
    Retrieval keeps prompt.txt primary, adding only a small ranked context slice.
    """
    global SOOPUB_CONTEXT_CACHE
    cache_key = f"{(query_text or '').strip().lower()}|{max_docs}|{max_chars}"
    if cache_key in SOOPUB_CONTEXT_CACHE:
        return SOOPUB_CONTEXT_CACHE[cache_key]

    docs = _load_soopub_docs()
    if not docs:
        SOOPUB_CONTEXT_CACHE[cache_key] = ""
        return ""

    query_tokens = _keyword_tokens(query_text)
    preferred = {
        'soopub/7dict/mantle-hood organogram system.md',
        'soopub/4int/sound motion object.md',
        'soopub/5pri/acoustic.md',
        'soopub/1mat/material-acoustic-atlas.md',
        'soopub/2obj/platonic-geometry-acoustics.md'
    }

    def score(doc):
        overlap = len(doc['tokens'] & query_tokens) if query_tokens else 0
        preferred_bonus = 3 if doc['label'] in preferred else 0
        return overlap * 5 + preferred_bonus

    ranked = sorted(docs, key=score, reverse=True)
    selected = ranked[:max_docs]

    chunks = []
    used_chars = 0
    for doc in selected:
        block = f"[{doc['label']}]\n{doc['text']}".strip()
        if not block:
            continue
        if used_chars + len(block) > max_chars and chunks:
            break
        chunks.append(block)
        used_chars += len(block)

    context = "\n\n".join(chunks).strip()
    SOOPUB_CONTEXT_CACHE[cache_key] = context
    return context


def _soopub_base_dir():
    return os.path.join(os.path.dirname(__file__), 'soopub')


def _collapse_text(value: str, max_chars: int = None) -> str:
    text = re.sub(r'\s+', ' ', (value or '')).strip()
    if max_chars and len(text) > max_chars:
        return text[:max_chars].rstrip() + '...'
    return text


def _extract_sketch_color_hints(plot_code: str):
    color_bank = [
        'beige', 'brown', 'pink', 'grey', 'gray', 'orange', 'yellow', 'lightgrey',
        'green', 'white', 'blue', 'black', 'red', 'purple', 'cyan', 'magenta',
        'teal', 'amber', 'gold', 'silver'
    ]
    lowered = (plot_code or '').lower()
    colors = []
    for color in color_bank:
        if color in lowered and color not in colors:
            colors.append(color)
    for hex_match in re.findall(r'#[0-9a-fA-F]{6}', plot_code or ''):
        if hex_match not in colors:
            colors.append(hex_match)
    return colors[:8]


def _extract_sketch_shape_hints(plot_code: str):
    lowered = (plot_code or '').lower()
    shape_pairs = [
        ('circle', 'circular resonators'),
        ('rectangle', 'rectilinear frames'),
        ('ellipse', 'elliptical chambers'),
        ('polygon', 'faceted polygons'),
        ('regularpolygon', 'polygonal nodes'),
        ('fancyarrowpatch', 'directional arrows'),
        ('annotate', 'gesture arrows'),
        ('arc', 'semi-circular markers'),
        ('linestring', 'tube-like paths')
    ]
    shapes = []
    for token, label in shape_pairs:
        if token in lowered and label not in shapes:
            shapes.append(label)
    return shapes[:6]


def build_sketch_prompt(
    prompt: str,
    prompt_content: str = "",
    summary_text: str = "",
    materials_text: str = "",
    plot_code: str = ""
) -> str:
    """
    Constructs a concise image generation prompt optimized for CLIP (77 token limit).
    Prioritizes black background and the first lines of the summary.
    """
    shape_hints = _extract_sketch_shape_hints(plot_code)
    color_hints = _extract_sketch_color_hints(plot_code)

    # 1. PRIMARY IDENTITY (Highest priority: first sentence of summary)
    summary_start = ""
    if summary_text:
        # Extract first two meaningful sentences or first 140 chars
        sentences = [s.strip() for s in re.split(r'[.!?]+', summary_text) if s.strip()]
        summary_start = ". ".join(sentences[:2]).strip()
        summary_start = _collapse_text(summary_start, max_chars=140)

    # 2. USER CONCEPT (Secondary priority)
    user_concept = _collapse_text(prompt, max_chars=80) if prompt else ""
    
    # 3. GEOMETRY (Limited tokens to avoid overriding summary)
    geom = ", ".join(shape_hints[:3]) if shape_hints else ""
    colors = ", ".join(color_hints[:4]) if color_hints else ""

    # 4. BUILD CONCISE PROMPT
    # Structure: [Background] [Identity] [Context] [Style] [Form/Color]
    parts = [
        "solid black background, obsidian backdrop.",
        f"A speculative instrument: {summary_start}." if summary_start else "A futuristic musical instrument.",
        f"Inspired by: {user_concept}." if user_concept else "",
        "realistic 3D object, highly imaginative, cinematic lighting, dark mode.",
        f"Form: {geom}. Colors: {colors}." if geom or colors else "",
        "photorealistic rendering, hyper-detailed, 8k resolution, black background."
    ]
    
    full_prompt = " ".join([p for p in parts if p])
    return _collapse_text(full_prompt, max_chars=400)


def _load_sketch_pipeline(model_id: str, cache_pipeline: bool = True):
    global SKETCH_PIPELINE, SKETCH_PIPELINE_MODEL

    if cache_pipeline and SKETCH_PIPELINE is not None and SKETCH_PIPELINE_MODEL == model_id:
        return SKETCH_PIPELINE

    from diffusers import AutoPipelineForImage2Image

    torch_dtype = torch.float16 if device.type == 'cuda' else torch.float32
    base_kwargs = {
        'torch_dtype': torch_dtype,
        'cache_dir': HF_CACHE_DIR
    }
    load_variants = [
        {'safety_checker': None, 'requires_safety_checker': False, **base_kwargs},
        {'safety_checker': None, **base_kwargs},
        base_kwargs
    ]

    pipe = None
    last_error = None
    for kwargs in load_variants:
        try:
            pipe = AutoPipelineForImage2Image.from_pretrained(model_id, **kwargs)
            break
        except TypeError as e:
            last_error = e
            continue
        except Exception as e:
            last_error = e
            break

    if pipe is None:
        raise RuntimeError(f"Could not load sketch pipeline {model_id}: {last_error}")

    pipe = pipe.to(device)
    if hasattr(pipe, 'enable_attention_slicing'):
        pipe.enable_attention_slicing()
    
    # Updated to non-deprecated VAE slicing call
    if hasattr(pipe, 'vae') and hasattr(pipe.vae, 'enable_slicing'):
        pipe.vae.enable_slicing()
    elif hasattr(pipe, 'enable_vae_slicing'):
        pipe.enable_vae_slicing()

    if hasattr(pipe, 'set_progress_bar_config'):
        pipe.set_progress_bar_config(disable=True)

    if cache_pipeline:
        SKETCH_PIPELINE = pipe
        SKETCH_PIPELINE_MODEL = model_id

    return pipe


def generate_sketch_image(
    organogram_bytes: bytes,
    prompt: str,
    prompt_content: str = "",
    summary_text: str = "",
    materials_text: str = "",
    plot_code: str = ""
):
    config = get_sketch_config()
    if not config.get('enabled'):
        return None, None, None
    if not organogram_bytes:
        return None, None, None

    sketch_prompt_content = get_specialized_prompt('inferred-image')
    sketch_prompt = build_sketch_prompt(
        prompt=prompt,
        prompt_content=sketch_prompt_content,
        summary_text=summary_text,
        materials_text=materials_text,
        plot_code=plot_code
    )
    negative_prompt = (
        "white background, light background, bright background, grey background, "
        "text, typography, labels, caption, watermark, graph, chart, axes, legend, ui, "
        "abstract infographic, collage, multiple instruments, photorealistic photo"
    )

    pipe = _load_sketch_pipeline(
        model_id=config['model_id'],
        cache_pipeline=bool(config.get('cache_pipeline'))
    )

    organogram_image = Image.open(BytesIO(organogram_bytes)).convert('RGB')
    organogram_image = organogram_image.resize((config['width'], config['height']), Image.LANCZOS)

    seed_value = int(hashlib.sha256(f"{prompt}|{summary_text}|{materials_text}".encode('utf-8')).hexdigest()[:8], 16)
    generator_device = 'cuda' if device.type == 'cuda' else 'cpu'
    generator = torch.Generator(device=generator_device).manual_seed(seed_value)

    with torch.inference_mode():
        result = pipe(
            prompt=sketch_prompt,
            negative_prompt=negative_prompt,
            image=organogram_image,
            strength=float(config['strength']),
            guidance_scale=float(config['guidance_scale']),
            num_inference_steps=int(config['steps']),
            generator=generator
        )

    sketch_image = result.images[0].convert('RGB')
    output = BytesIO()
    sketch_image.save(output, format='PNG', optimize=True)

    if not config.get('cache_pipeline'):
        try:
            pipe.to('cpu')
            del pipe
        except Exception:
            pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc
        gc.collect()

    return output.getvalue(), sketch_prompt, config['model_id']


def _load_sound_pipeline(model_id: str, cache_pipeline: bool = True):
    global SOUND_PIPELINE, SOUND_PIPELINE_MODEL

    if cache_pipeline and SOUND_PIPELINE is not None and SOUND_PIPELINE_MODEL == model_id:
        return SOUND_PIPELINE

    from diffusers import StableAudioPipeline

    token = os.getenv('HF_TOKEN') or os.getenv('STABLE_AUDIO_OPEN_TOKEN')
    logging.info(f"Loading Stable Audio pipeline: {model_id} (Token: {'provided' if token else 'NOT provided'})")
    
    torch_dtype = torch.float16 if device.type in ('cuda', 'mps') else torch.float32
    logging.info(f"Using torch_dtype: {torch_dtype} on device: {device.type}")
    
    try:
        logging.info("Starting from_pretrained...")
        pipe = StableAudioPipeline.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            token=token,
            cache_dir=HF_CACHE_DIR,
            trust_remote_code=True
        )
        logging.info("Pipeline loaded successfully into RAM. Moving to device...")
        pipe = pipe.to(device)
        logging.info(f"Pipeline ready on {device.type}.")
    except Exception as e:
        logging.error(f"Failed to load Stable Audio pipeline: {e}")
        raise

    if cache_pipeline:
        SOUND_PIPELINE = pipe
        SOUND_PIPELINE_MODEL = model_id

    return pipe


def get_audio_config():
    return {
        'model_id': (os.getenv('SOOG_SOUND_MODEL', 'stabilityai/stable-audio-open-1.0') or '').strip() or 'stabilityai/stable-audio-open-1.0',
        'duration_sec': _env_int('SOOG_SOUND_DURATION_SEC', 15, min_value=1, max_value=47),
        'steps': _env_int('SOOG_SOUND_STEPS', 100, min_value=10, max_value=500),
        'cache_pipeline': _env_bool('SOOG_SOUND_CACHE_PIPELINE', default=True)
    }


def build_audio_prompts(prompt: str, summary_text: str, materials_text: str, request_id: str = None):
    """
    Use an LLM to generate rich audio prompts based on organogram data and prompt_sound.txt instructions.
    Returns a list of 3-5 strings.
    """
    sound_instructions = get_specialized_prompt('sound')
    
    system_prompt = (
        "You are an expert sound designer specializing in timbre and isolated sound objects. "
        "Your task is to transform a conceptual instrument description into highly precise "
        "audio generation prompts for Stable Audio Open."
    )
    
    user_prompt = (
        f"Following these instructions:\n\n{sound_instructions}\n\n"
        f"Design 3 unique and varied audio prompts that capture different timbral possibilities of this instrument:\n"
        f"User Concept: {prompt}\n"
        f"Conceptual Summary: {summary_text}\n"
        f"Materials: {materials_text}\n\n"
        "Return exactly 3 lines, one for each prompt. No numbering, no extra text. Each line should be a self-contained prompt."
    )
    
    update_generation_progress(request_id, stage='sound_prompt_design', status='running')
    
    def progress_cb(preview):
        update_generation_progress(request_id, stage='sound_prompt_design', status='running', reasoning_preview=preview)
        
    try:
        content, meta = call_ollama_chat(
            system_prompt,
            user_prompt,
            timeout=60,
            max_tokens=600,
            temperature=0.8,
            progress_callback=progress_cb if request_id else None
        )
        prompts = [p.strip() for p in content.splitlines() if p.strip()]
        # Clean up any leading markers like "1. ", "- ", etc.
        prompts = [re.sub(r'^[\d\.\-\s\*]+', '', p) for p in prompts]
        return prompts[:3]
    except Exception as e:
        logging.error(f"Error generating audio prompts: {e}")
        # Simple fallback
        return [f"isolated sound object, {summary_text[:120]}, high quality"] * 3


def convert_to_ogg(wav_path: str):
    """
    Convert a wav file to ogg for web performance.
    """
    ogg_path = wav_path.replace('.wav', '.ogg')
    try:
        import subprocess
        # -y to overwrite, -acodec libopus or just default ogg
        # using libopus for better quality/size if available, but -acodec libvorbis is more standard for .ogg
        cmd = ['ffmpeg', '-y', '-i', wav_path, '-acodec', 'libvorbis', ogg_path]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return ogg_path
    except Exception as e:
        logging.error(f"Failed to convert {wav_path} to ogg: {e}")
        return None


def generate_sound_samples(prompt: str, summary_text: str, materials_text: str, basename: str, request_id: str = None):
    config = get_audio_config()
    
    audio_prompts = build_audio_prompts(prompt, summary_text, materials_text, request_id=request_id)
    
    update_generation_progress(request_id, stage='loading_sound_pipeline', status='running', reasoning_preview="Loading Stable Audio (this can take several minutes if weights are downloading)...")
    try:
        pipe = _load_sound_pipeline(
            model_id=config['model_id'],
            cache_pipeline=bool(config.get('cache_pipeline'))
        )
    except Exception as e:
        update_generation_progress(request_id, stage='error', status='error', reasoning_preview=f"Failed to load sound pipeline: {str(e)}")
        raise
    
    results = []
    rate = pipe.vae.sampling_rate if hasattr(pipe, 'vae') else 44100
    
    for i, audio_prompt in enumerate(audio_prompts):
        msg = f"Generating sample {i+1}/{len(audio_prompts)}: {audio_prompt[:80]}..."
        update_generation_progress(request_id, stage='sound_generation', status='running', reasoning_preview=msg)
        
        seed_value = int(hashlib.sha256(f"{audio_prompt}|{i}|{basename}".encode('utf-8')).hexdigest()[:8], 16)
        
        # Generator device handling
        gen_device = 'cpu' if device.type == 'mps' else device.type
        generator = torch.Generator(device=gen_device).manual_seed(seed_value)
        
        try:
            with torch.inference_mode():
                output = pipe(
                    prompt=audio_prompt,
                    num_inference_steps=config['steps'],
                    audio_end_in_s=float(config['duration_sec']),
                    generator=generator
                )
            
            audio_data = output.audios[0] # Usually [num_channels, num_samples]
            
            # Convert to numpy if it's a tensor
            if torch.is_tensor(audio_data):
                audio_data = audio_data.cpu().numpy()
            
            # Ensure it's [samples, channels] for scipy.io.wavfile
            if audio_data.ndim == 2:
                # If [channels, samples], transpose it
                if audio_data.shape[0] < audio_data.shape[1] and audio_data.shape[0] in (1, 2):
                    audio_data = audio_data.T
            
            sample_filename = f"{basename}_sample{i+1}.wav"
            sample_path = os.path.join(GALLERY_DIR, sample_filename)
            
            scipy.io.wavfile.write(sample_path, rate, audio_data)
            
            # Also convert to ogg
            update_generation_progress(request_id, stage='converting_to_ogg', status='running', reasoning_preview=f"Optimizing sample {i+1} for web...")
            ogg_path = convert_to_ogg(sample_path)
            ogg_url = None
            if ogg_path:
                ogg_url = f"/api/gallery/image/{os.path.basename(ogg_path)}"
            
            results.append({
                'url': f"/api/gallery/image/{sample_filename}",
                'ogg_url': ogg_url,
                'prompt': audio_prompt
            })
        except Exception as e:
            logging.error(f"Error generating sample {i+1}: {e}")
            continue
    
    if not results:
        update_generation_progress(request_id, stage='error', status='error', reasoning_preview="No sound samples were generated successfully.")
    else:
        update_generation_progress(request_id, stage='completed', status='completed')
        
    if not config.get('cache_pipeline'):
        try:
            pipe.to('cpu')
            del pipe
        except Exception:
            pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc
        gc.collect()
        
    return results, config['model_id']


def _normalize_graph_token(value: str) -> str:
    token = (value or '').strip().lower().replace('\\', '/')
    token = re.sub(r'\.[a-z0-9]+$', '', token)
    token = re.sub(r'[/\s]+', ' ', token)
    token = re.sub(r'[^a-z0-9 _\-/]+', '', token)
    token = re.sub(r'\s+', ' ', token).strip()
    return token


def _split_frontmatter(content: str):
    if not content or not content.startswith('---'):
        return "", content
    lines = content.splitlines()
    if not lines or lines[0].strip() != '---':
        return "", content
    for idx in range(1, len(lines)):
        if lines[idx].strip() == '---':
            frontmatter = "\n".join(lines[1:idx])
            body = "\n".join(lines[idx + 1:])
            return frontmatter, body
    return "", content


def _strip_yaml_scalar(value: str) -> str:
    text = (value or '').strip().strip('"\'')
    if text.startswith('[[') and text.endswith(']]'):
        text = text[2:-2]
    if text.startswith('#'):
        text = text[1:]
    return text.strip()


def _parse_frontmatter_tags(frontmatter: str):
    if not frontmatter:
        return []
    tags = []
    seen = set()
    lines = frontmatter.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        match = re.match(r'^\s*tags\s*:\s*(.*)$', line, flags=re.IGNORECASE)
        if not match:
            i += 1
            continue

        remainder = (match.group(1) or '').strip()
        if remainder:
            remainder = remainder.strip('[]')
            for token in re.split(r'[,\s]+', remainder):
                tag = _strip_yaml_scalar(token)
                normalized = _normalize_graph_token(tag)
                if normalized and normalized not in seen:
                    seen.add(normalized)
                    tags.append(normalized)
            i += 1
            continue

        i += 1
        while i < len(lines):
            row = lines[i]
            item = re.match(r'^\s*-\s*(.+)$', row)
            if item:
                tag = _strip_yaml_scalar(item.group(1))
                normalized = _normalize_graph_token(tag)
                if normalized and normalized not in seen:
                    seen.add(normalized)
                    tags.append(normalized)
                i += 1
                continue
            if not row.strip():
                i += 1
                continue
            break
    return tags


def _extract_inline_tags(body: str):
    tags = []
    seen = set()
    for line in (body or '').splitlines():
        if re.match(r'^\s{0,3}#{1,6}\s', line):
            continue
        for match in re.finditer(r'(?<![\w/])#([a-zA-Z0-9][a-zA-Z0-9_/\-]{1,63})', line):
            tag = _normalize_graph_token(match.group(1))
            if tag and tag not in seen:
                seen.add(tag)
                tags.append(tag)
    return tags


def _extract_wikilinks(content: str):
    links = []
    seen = set()
    for raw in re.findall(r'\[\[([^\]]+)\]\]', content or ''):
        cleaned = raw.strip()
        if not cleaned:
            continue
        if cleaned not in seen:
            seen.add(cleaned)
            links.append(cleaned)
    return links


def _normalize_wikilink_target(value: str) -> str:
    target = (value or '').strip()
    if not target:
        return ''
    target = target.split('|', 1)[0].split('#', 1)[0].strip()
    target = target.replace('\\', '/')
    target = re.sub(r'\.md$', '', target, flags=re.IGNORECASE)
    return target.strip()


def _soopub_files_signature(base_dir: str):
    parts = []
    for root, _, files in os.walk(base_dir):
        for filename in sorted(files):
            if filename.startswith('.'):
                continue
            if not filename.lower().endswith(('.md', '.txt')):
                continue
            fp = os.path.join(root, filename)
            try:
                stat = os.stat(fp)
            except OSError:
                continue
            rel = os.path.relpath(fp, base_dir).replace('\\', '/')
            parts.append(f"{rel}:{int(stat.st_mtime)}:{stat.st_size}")
    return "|".join(parts)


def _build_soopub_graph():
    base_dir = _soopub_base_dir()
    if not os.path.isdir(base_dir):
        return {'nodes': [], 'links': [], 'stats': {'nodes': 0, 'links': 0}}

    signature = _soopub_files_signature(base_dir)
    cached = SOOPUB_GRAPH_CACHE.get('snapshot')
    if cached and cached.get('signature') == signature:
        return cached.get('graph')

    nodes = {}
    links = []
    link_keys = set()
    note_records = []

    def add_node(payload: dict):
        node_id = payload.get('id')
        if not node_id:
            return
        if node_id in nodes:
            return
        nodes[node_id] = payload

    def add_link(source: str, target: str, edge_type: str):
        if not source or not target or source == target:
            return
        key = (source, target, edge_type)
        if key in link_keys:
            return
        link_keys.add(key)
        links.append({
            'source': source,
            'target': target,
            'type': edge_type
        })

    root_folder_id = 'folder:soopub'
    add_node({
        'id': root_folder_id,
        'type': 'folder',
        'title': 'soopub',
        'path': 'soopub',
        'section_key': 'root',
        'section_label': 'vault',
        'depth': 0
    })

    for root, dirs, files in os.walk(base_dir):
        dirs[:] = sorted([d for d in dirs if not d.startswith('.')])
        files = sorted(files)

        rel_dir = os.path.relpath(root, base_dir).replace('\\', '/')
        rel_dir = '' if rel_dir == '.' else rel_dir

        folder_id = root_folder_id if not rel_dir else f"folder:{rel_dir}"
        folder_name = 'soopub' if not rel_dir else os.path.basename(root)
        folder_depth = 0 if not rel_dir else rel_dir.count('/') + 1
        top_section = rel_dir.split('/')[0] if rel_dir else 'root'
        top_label = SOOPUB_SECTION_LABELS.get(top_section, top_section)

        folder_title = folder_name
        if rel_dir and rel_dir.count('/') == 0 and top_section in SOOPUB_SECTION_LABELS:
            folder_title = f"{folder_name} ({top_label})"

        add_node({
            'id': folder_id,
            'type': 'folder',
            'title': folder_title,
            'path': rel_dir or 'soopub',
            'section_key': top_section,
            'section_label': top_label,
            'depth': folder_depth
        })

        if rel_dir:
            parent_rel = os.path.dirname(rel_dir).replace('\\', '/')
            parent_id = root_folder_id if not parent_rel else f"folder:{parent_rel}"
            add_link(parent_id, folder_id, 'folder_contains')

        for filename in files:
            if filename.startswith('.'):
                continue
            if not filename.lower().endswith(('.md', '.txt')):
                continue

            fp = os.path.join(root, filename)
            rel_file = os.path.relpath(fp, base_dir).replace('\\', '/')
            rel_no_ext = re.sub(r'\.[^.]+$', '', rel_file)
            title = os.path.splitext(os.path.basename(filename))[0].strip()
            note_id = f"note:{rel_no_ext}"
            section_key = rel_file.split('/')[0] if '/' in rel_file else 'root'
            section_label = SOOPUB_SECTION_LABELS.get(section_key, section_key)

            try:
                with open(fp, 'r', encoding='utf-8') as f:
                    content = f.read()
            except Exception:
                continue

            frontmatter, body = _split_frontmatter(content)
            preview = re.sub(r'\s+', ' ', (body or '').strip())[:360]
            tags = []
            tag_seen = set()
            for tag in _parse_frontmatter_tags(frontmatter) + _extract_inline_tags(body):
                normalized = _normalize_graph_token(tag)
                if normalized and normalized not in tag_seen:
                    tag_seen.add(normalized)
                    tags.append(normalized)
            wikilinks = _extract_wikilinks(content)

            add_node({
                'id': note_id,
                'type': 'note',
                'title': title or rel_no_ext,
                'path': rel_file,
                'section_key': section_key,
                'section_label': section_label,
                'folder_id': folder_id,
                'preview': preview,
                'tag_count': len(tags)
            })
            add_link(folder_id, note_id, 'folder_contains')

            note_records.append({
                'id': note_id,
                'title': title,
                'path_no_ext': rel_no_ext,
                'wikilinks': wikilinks,
                'tags': tags
            })

    note_lookup = {}
    for record in note_records:
        keys = {
            _normalize_graph_token(record['title']),
            _normalize_graph_token(record['path_no_ext']),
            _normalize_graph_token(os.path.basename(record['path_no_ext'])),
            _normalize_graph_token(record['path_no_ext'].replace('/', ' '))
        }
        for key in keys:
            if key and key not in note_lookup:
                note_lookup[key] = record['id']

    for record in note_records:
        note_id = record['id']

        for tag in record['tags']:
            tag_id = f"tag:{tag}"
            tag_leaf = tag.split('/')[-1]
            add_node({
                'id': tag_id,
                'type': 'tag',
                'title': f"#{tag_leaf}",
                'path': tag,
                'section_key': 'tag',
                'section_label': 'tags'
            })
            add_link(note_id, tag_id, 'tagged')

        for raw_target in record['wikilinks']:
            target = _normalize_wikilink_target(raw_target)
            if not target:
                continue

            target_key = _normalize_graph_token(target)
            target_basename = _normalize_graph_token(os.path.basename(target))
            target_id = note_lookup.get(target_key) or note_lookup.get(target_basename)

            if target_id and target_id != note_id:
                add_link(note_id, target_id, 'wikilink')
                continue

            ghost_key = _normalize_graph_token(target)
            if not ghost_key:
                continue
            ghost_id = f"ghost:{ghost_key}"
            add_node({
                'id': ghost_id,
                'type': 'ghost',
                'title': target,
                'path': target,
                'section_key': 'reference',
                'section_label': 'external references'
            })
            add_link(note_id, ghost_id, 'wikilink_unresolved')

    degree = {node_id: 0 for node_id in nodes.keys()}
    for edge in links:
        source = edge.get('source')
        target = edge.get('target')
        if source in degree:
            degree[source] += 1
        if target in degree:
            degree[target] += 1

    for node in nodes.values():
        node_degree = degree.get(node['id'], 0)
        node['degree'] = node_degree
        if node.get('type') == 'folder':
            node['val'] = 3 + min(node_degree, 20) * 0.18
        elif node.get('type') == 'note':
            node['val'] = 2.4 + min(node_degree, 24) * 0.16
        elif node.get('type') == 'tag':
            node['val'] = 1.8 + min(node_degree, 18) * 0.14
        else:
            node['val'] = 1.6 + min(node_degree, 12) * 0.12

    graph = {
        'nodes': list(nodes.values()),
        'links': links,
        'stats': {
            'nodes': len(nodes),
            'links': len(links),
            'notes': len([n for n in nodes.values() if n.get('type') == 'note']),
            'folders': len([n for n in nodes.values() if n.get('type') == 'folder']),
            'tags': len([n for n in nodes.values() if n.get('type') == 'tag']),
            'ghosts': len([n for n in nodes.values() if n.get('type') == 'ghost'])
        },
        'section_labels': SOOPUB_SECTION_LABELS
    }
    SOOPUB_GRAPH_CACHE['snapshot'] = {'signature': signature, 'graph': graph}
    return graph


def _repair_matplotlib_imports(code: str) -> str:
    lines = code.splitlines()
    fixed = []
    needs_line2d = False

    for line in lines:
        stripped = line.strip()
        match = re.match(r'from\s+matplotlib\.patches\s+import\s+(.+)$', stripped)
        if not match:
            fixed.append(line)
            continue

        names = [name.strip() for name in match.group(1).split(',') if name.strip()]
        if 'Line2D' not in names:
            fixed.append(line)
            continue

        names = [name for name in names if name != 'Line2D']
        needs_line2d = True
        indent = line[:len(line) - len(line.lstrip())]
        if names:
            fixed.append(f"{indent}from matplotlib.patches import {', '.join(names)}")

    patched = "\n".join(fixed)
    if needs_line2d and not re.search(r'from\s+matplotlib\.lines\s+import\s+Line2D', patched):
        patched = f"from matplotlib.lines import Line2D\n{patched}"
    return patched


def _safe_slug(text: str, fallback: str = "item") -> str:
    slug = re.sub(r'[^a-z0-9]+', '_', (text or "").lower()).strip('_')
    return slug or fallback


def _version_index_from_value(value, default: int = 1) -> int:
    if value is None:
        return default
    if isinstance(value, int):
        return max(1, value)
    if isinstance(value, float):
        return max(1, int(round(value)))
    text = str(value).strip().lower()
    if not text:
        return default
    if text.startswith('v'):
        text = text[1:]
    plain = re.match(r'^(\d+)$', text)
    if plain:
        return max(1, int(plain.group(1)))
    legacy = re.match(r'^(\d+)\.(\d+)$', text)
    if legacy:
        major = int(legacy.group(1))
        minor = int(legacy.group(2))
        # Legacy v1.0/v1.1/v1.2... maps to 1/2/3...
        if major == 1:
            return max(1, minor + 1)
        return max(1, major * 10 + minor)
    return default


def _version_label_from_index(version_index: int) -> str:
    return f"v{max(1, int(version_index or 1))}"


def _title_from_basename(basename: str) -> str:
    if not basename:
        return "untitled"
    parts = basename.split('_')
    if len(parts) <= 1:
        return basename
    # drop timestamp prefix
    return parts[1].replace('_', '-')


def _basename_body(basename: str) -> str:
    value = str(basename or '').strip()
    if not value:
        return ''
    return re.sub(r'^\d{8}-\d{6}_', '', value)


def _infer_group_id_from_basename(basename: str) -> str:
    body = _basename_body(basename)
    if not body:
        return ''
    # Match both v1 and legacy v1_0 / v1_0_1 suffixes.
    stripped = re.sub(r'_v\d+(?:_\d+)?(?:_\d+)?$', '', body, flags=re.IGNORECASE)
    return stripped or body


def _meta_group_id(meta: dict) -> str:
    if not isinstance(meta, dict):
        return ''
    explicit = str(meta.get('group_id') or '').strip()
    if explicit:
        return explicit
    return _infer_group_id_from_basename(str(meta.get('basename') or '').strip())


def parse_refact_marker(prompt_text: str):
    """
    Parse optional first-line refactor markers:
    - [REFACT key=value key2=value2]
    - * ... (short refactor mode)
    Returns: (clean_prompt, meta_dict)
    """
    text = (prompt_text or "").strip()
    meta = {'is_refact': False}
    if not text:
        return "", meta

    lines = text.splitlines()
    first = lines[0].strip()

    if first.startswith('[REFACT'):
        meta['is_refact'] = True
        payload = first[len('[REFACT'):].rstrip(']').strip()
        for token in re.split(r'\s+', payload):
            if '=' not in token:
                continue
            key, val = token.split('=', 1)
            key = key.strip().lower()
            val = val.strip().strip('"\'')
            if key:
                meta[key] = val
        clean = "\n".join(lines[1:]).lstrip()
        return clean, meta

    if first.startswith('*') or first.startswith('+'):
        meta['is_refact'] = True
        clean_first = first[1:].strip()
        rest = "\n".join(lines[1:]).strip()
        clean = f"{clean_first}\n{rest}".strip()
        return clean, meta

    return text, meta


def _iter_gallery_metas():
    if not os.path.isdir(GALLERY_DIR):
        return []
    metas = []
    for name in os.listdir(GALLERY_DIR):
        if not name.endswith('.json'):
            continue
        path = os.path.join(GALLERY_DIR, name)
        try:
            with open(path, 'r', encoding='utf-8') as f:
                metas.append(json.load(f))
        except Exception:
            continue
    return metas


def _group_defaults(group_id: str):
    title = None
    title_slug = None
    max_version_index = 0
    for meta in _iter_gallery_metas():
        if _meta_group_id(meta) != group_id:
            continue
        if title is None:
            title = meta.get('title')
        if title_slug is None:
            title_slug = meta.get('title_slug')
        max_version_index = max(
            max_version_index,
            _version_index_from_value(meta.get('version_index'), default=0)
        )
    return {'title': title, 'title_slug': title_slug, 'max_version_index': max_version_index}


def _next_group_version_index(group_id: str, seed_version_index: int = 1) -> int:
    group_items = [meta for meta in _iter_gallery_metas() if _meta_group_id(meta) == group_id]
    if not group_items:
        return max(1, seed_version_index)

    # Prefer explicit integer versions (v1, v2, v3).
    max_plain = 0
    for meta in group_items:
        version_text = str(meta.get('version') or '').strip().lower()
        match = re.match(r'^v?(\d+)$', version_text)
        if match:
            max_plain = max(max_plain, int(match.group(1)))
            continue
        idx_value = meta.get('version_index')
        if isinstance(idx_value, int) and 1 <= idx_value < 10:
            max_plain = max(max_plain, idx_value)

    if max_plain > 0:
        return max(max_plain + 1, max(1, seed_version_index))

    # Legacy v1.0/v1.1 data: continue with simple ordinal count.
    return len(group_items) + 1


def extract_llm_content(payload: dict) -> str:
    # OpenAI-compatible format
    choices = payload.get('choices')
    if isinstance(choices, list) and choices:
        message = choices[0].get('message', {})
        content = message.get('content')
        if isinstance(content, list):
            text_parts = [
                part.get('text', '')
                for part in content
                if isinstance(part, dict)
            ]
            text = ''.join(text_parts).strip()
            if text:
                return text
        elif content is not None:
            return str(content).strip()

    # Native Ollama /api/chat format
    message = payload.get('message')
    if isinstance(message, dict) and message.get('content') is not None:
        return str(message.get('content')).strip()

    # Native Ollama /api/generate format
    if payload.get('response') is not None:
        return str(payload.get('response')).strip()

    raise ValueError('Response does not contain text content')


def extract_llm_thinking(payload: dict) -> str:
    message = payload.get('message')
    if isinstance(message, dict) and message.get('thinking') is not None:
        return str(message.get('thinking') or '')
    if payload.get('thinking') is not None:
        return str(payload.get('thinking') or '')
    return ''


def stream_ollama_response(url: str, headers: dict, body: dict, timeout=90, progress_callback=None):
    response = requests.post(url, headers=headers, json=body, timeout=timeout, stream=True)

    if response.status_code >= 400:
        response_text = response.text[:600]
        if response.status_code in (400, 404, 405, 422):
            raise RuntimeError(f"Endpoint {url} returned {response.status_code}: {response_text}")
        if response.status_code == 401:
            raise RuntimeError(f"Ollama unauthorized at {url}. Check OLLAMA_API_KEY or proxy auth.")
        if 'requires more system memory' in response_text.lower():
            raise RuntimeError(
                f"Ollama model OOM at {url}: {response_text}. Try a smaller model or free RAM/swap."
            )
        raise RuntimeError(f"Ollama API error ({response.status_code}) at {url}: {response_text}")

    content_parts = []
    thinking_parts = []
    final_payload = None

    for raw_line in response.iter_lines(decode_unicode=True):
        if not raw_line:
            continue
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except Exception:
            continue

        final_payload = payload
        thinking_piece = extract_llm_thinking(payload)
        if thinking_piece:
            thinking_parts.append(thinking_piece)
            if progress_callback:
                progress_callback(_short_reasoning_preview("".join(thinking_parts)))

        message = payload.get('message')
        if isinstance(message, dict) and message.get('content') is not None:
            content_parts.append(str(message.get('content') or ''))
        elif payload.get('response') is not None:
            content_parts.append(str(payload.get('response') or ''))

    content = ''.join(content_parts).strip()
    thinking = ''.join(thinking_parts).strip()
    if not content and final_payload is not None:
        content = extract_llm_content(final_payload)
    return content, thinking, final_payload or {}


def _extract_json_object(text: str):
    raw = (text or '').strip()
    if not raw:
        raise ValueError('Empty JSON content')
    raw = re.sub(r'^```(?:json)?\s*', '', raw, flags=re.IGNORECASE)
    raw = re.sub(r'\s*```$', '', raw)
    try:
        return json.loads(raw)
    except Exception:
        match = re.search(r'(\{[\s\S]*\})', raw)
        if match:
            return json.loads(match.group(1))
        raise


def build_plot_generation_schema():
    return {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "materials": {"type": "string"},
            "plot_code": {"type": "string"}
        },
        "required": ["summary", "materials", "plot_code"]
    }


def build_stl_generation_schema():
    return {
        "type": "object",
        "properties": {
            "stl_code": {"type": "string"}
        },
        "required": ["stl_code"]
    }


def call_ollama_structured(
    system_prompt: str,
    user_prompt: str,
    schema: dict,
    model_name_override: str = None,
    timeout=90,
    max_tokens: int = 1000,
    temperature: float = 0.2,
    progress_callback=None
):
    base_url, model_name, _ = get_ollama_config()
    model_name = model_name_override or model_name
    headers = get_ollama_headers()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    def build_attempts(think_enabled: bool):
        native_body = {
            "model": model_name,
            "messages": messages,
            "stream": True,
            "format": schema,
            "think": think_enabled,
            "options": {
                "temperature": float(temperature),
                "num_predict": int(max_tokens)
            }
        }
        openai_body = {
            "model": model_name,
            "messages": messages,
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "soog_generation",
                    "schema": schema
                }
            }
        }
        generate_body = {
            "model": model_name,
            "system": system_prompt,
            "prompt": user_prompt,
            "stream": True,
            "format": schema,
            "think": think_enabled,
            "options": {
                "temperature": float(temperature),
                "num_predict": int(max_tokens)
            }
        }
        return [
            (f"{base_url}/api/chat", native_body),
            (f"{base_url}/v1/chat/completions", openai_body),
            (f"{base_url}/api/generate", generate_body),
        ]

    last_error = None
    support_modes = [ollama_thinking_enabled(), False]
    seen_modes = set()

    for think_enabled in support_modes:
        if think_enabled in seen_modes:
            continue
        seen_modes.add(think_enabled)
        for url, body in build_attempts(think_enabled):
            try:
                if url.endswith('/api/chat') or url.endswith('/api/generate'):
                    content, thinking, _ = stream_ollama_response(
                        url,
                        headers=headers,
                        body=body,
                        timeout=timeout,
                        progress_callback=progress_callback
                    )
                    data = _extract_json_object(content)
                    return data, {
                        'endpoint': url,
                        'base_url': base_url,
                        'model': model_name,
                        'thinking': thinking
                    }

                response = requests.post(url, headers=headers, json=body, timeout=timeout)
                if response.status_code >= 400:
                    response_text = response.text[:600]
                    if think_enabled and 'does not support thinking' in response_text.lower():
                        last_error = response_text
                        break
                    if response.status_code in (400, 404, 405, 422):
                        last_error = f"Endpoint {url} returned {response.status_code}: {response_text}"
                        continue
                    if response.status_code == 401:
                        raise RuntimeError(
                            f"Ollama unauthorized at {url}. Check OLLAMA_API_KEY or proxy auth."
                        )
                    if 'requires more system memory' in response_text.lower():
                        raise RuntimeError(
                            f"Ollama model OOM at {url}: {response_text}. "
                            "Try a smaller model or free RAM/swap."
                        )
                    raise RuntimeError(f"Ollama API error ({response.status_code}) at {url}: {response_text}")

                payload = response.json()
                content = extract_llm_content(payload)
                data = _extract_json_object(content)
                return data, {
                    'endpoint': url,
                    'base_url': base_url,
                    'model': model_name,
                    'thinking': extract_llm_thinking(payload)
                }
            except requests.exceptions.RequestException as req_err:
                last_error = f"Request failed for {url}: {req_err}"
                continue
            except Exception as parse_err:
                error_text = str(parse_err)
                if think_enabled and 'does not support thinking' in error_text.lower():
                    last_error = error_text
                    break
                last_error = f"Invalid structured response from {url}: {parse_err}"
                continue

    raise RuntimeError(last_error or "Could not connect to Ollama structured endpoints")


def call_ollama_chat(
    system_prompt: str,
    user_prompt: str,
    model_name_override: str = None,
    timeout=90,
    max_tokens: int = 1000,
    temperature: float = 0.7,
    progress_callback=None
):
    base_url, model_name, _ = get_ollama_config()
    model_name = model_name_override or model_name
    headers = get_ollama_headers()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    def build_attempts(think_enabled: bool):
        openai_body = {
            "model": model_name,
            "messages": messages,
            "temperature": float(temperature),
            "max_tokens": int(max_tokens)
        }
        native_body = {
            "model": model_name,
            "messages": messages,
            "stream": True,
            "think": think_enabled,
            "options": {
                "temperature": float(temperature),
                "num_predict": int(max_tokens)
            }
        }
        generate_body = {
            "model": model_name,
            "prompt": f"System:\n{system_prompt}\n\nUser:\n{user_prompt}",
            "stream": True,
            "think": think_enabled,
            "options": {
                "temperature": float(temperature),
                "num_predict": int(max_tokens)
            }
        }
        return [
            (f"{base_url}/api/chat", native_body),
            (f"{base_url}/v1/chat/completions", openai_body),
            (f"{base_url}/api/generate", generate_body),
        ]

    last_error = None
    support_modes = [ollama_thinking_enabled(), False]
    seen_modes = set()

    for think_enabled in support_modes:
        if think_enabled in seen_modes:
            continue
        seen_modes.add(think_enabled)
        for url, body in build_attempts(think_enabled):
            try:
                if url.endswith('/api/chat') or url.endswith('/api/generate'):
                    content, thinking, _ = stream_ollama_response(
                        url,
                        headers=headers,
                        body=body,
                        timeout=timeout,
                        progress_callback=progress_callback
                    )
                    if not content:
                        last_error = f"Empty model response from {url}"
                        continue
                    return content, {
                        'endpoint': url,
                        'base_url': base_url,
                        'model': model_name,
                        'thinking': thinking
                    }

                response = requests.post(url, headers=headers, json=body, timeout=timeout)
                if response.status_code >= 400:
                    response_text = response.text[:600]
                    if think_enabled and 'does not support thinking' in response_text.lower():
                        last_error = response_text
                        break
                    if response.status_code in (400, 404, 405, 422):
                        last_error = f"Endpoint {url} returned {response.status_code}: {response_text}"
                        continue
                    if response.status_code == 401:
                        raise RuntimeError(
                            f"Ollama unauthorized at {url}. Check OLLAMA_API_KEY or proxy auth."
                        )
                    if 'requires more system memory' in response_text.lower():
                        raise RuntimeError(
                            f"Ollama model OOM at {url}: {response_text}. "
                            "Try a smaller model or free RAM/swap."
                        )
                    raise RuntimeError(f"Ollama API error ({response.status_code}) at {url}: {response_text}")

                payload = response.json()
                content = extract_llm_content(payload)
                if not content:
                    last_error = f"Empty model response from {url}"
                    continue
                return content, {
                    'endpoint': url,
                    'base_url': base_url,
                    'model': model_name,
                    'thinking': extract_llm_thinking(payload)
                }
            except requests.exceptions.RequestException as req_err:
                last_error = f"Request failed for {url}: {req_err}"
                continue
            except Exception as err:
                error_text = str(err)
                if think_enabled and 'does not support thinking' in error_text.lower():
                    last_error = error_text
                    break
                last_error = error_text
                continue

    raise RuntimeError(last_error or "Could not connect to Ollama endpoints")


def extract_python_code_blocks(raw_text: str):
    # First pass: closed fenced python blocks
    code_blocks = re.findall(r"```(?:python|py)?\s*([\s\S]*?)```", raw_text, re.DOTALL | re.IGNORECASE)

    # Fallback: detect opening fence without closure and grab to end
    if not code_blocks:
        m = re.search(r"```(?:python|py)?\s*([\s\S]*)$", raw_text, re.IGNORECASE)
        if m:
            code_blocks = [m.group(1)]

    # Last resort: heuristic when model emits bare code
    if not code_blocks and (('import matplotlib' in raw_text) or ('plt.' in raw_text)):
        start_idx = raw_text.find('import matplotlib')
        if start_idx == -1:
            start_idx = raw_text.find('plt.')
        if start_idx != -1:
            code_blocks = [raw_text[start_idx:]]

    return [c.strip() for c in code_blocks if c and c.strip()]


def _looks_like_plot_code(code: str) -> bool:
    lowered = (code or '').lower()
    plot_signals = [
        'matplotlib',
        'plt.',
        'subplots(',
        'fancyarrowpatch',
        'circle(',
        'rectangle(',
        'polygon(',
        'ellipse(',
        'regularpolygon(',
        'arc(',
        'ax.annotate',
        'line2d',
        'ax.add_patch'
    ]
    return any(signal in lowered for signal in plot_signals)


def _looks_like_stl_code(code: str) -> bool:
    lowered = (code or '').lower()
    stl_signals = [
        'trimesh',
        'import trimesh',
        'mesh =',
        'creation.cylinder',
        'creation.box',
        'creation.cone',
        'trimesh.creation',
        'trimesh.util.concatenate'
    ]
    return any(signal in lowered for signal in stl_signals)


def split_plot_and_stl_code(code_blocks):
    plot_code = None
    stl_code = None
    for idx, code in enumerate(code_blocks):
        if plot_code is None and _looks_like_plot_code(code) and not _looks_like_stl_code(code):
            plot_code = code
        if stl_code is None and _looks_like_stl_code(code):
            stl_code = code

    if plot_code is None:
        non_stl_blocks = [code for code in code_blocks if code != stl_code]
        if len(code_blocks) >= 2 and non_stl_blocks:
            plot_code = non_stl_blocks[0]
        elif len(code_blocks) == 1 and not _looks_like_stl_code(code_blocks[0]):
            plot_code = code_blocks[0]
    return plot_code, stl_code


def build_plot_output_template():
    return (
        "Return exactly in this order:\n"
        "Conceptual Summary\n"
        "<120-180 words>\n\n"
        "Materials\n"
        "1. <item>\n"
        "2. <item>\n\n"
        "```python\n"
        "import matplotlib.pyplot as plt\n"
        "import numpy as np\n"
        "from matplotlib.patches import Circle, Rectangle, Polygon, FancyArrowPatch\n"
        "plt.style.use('dark_background')\n"
        "fig, ax = plt.subplots(figsize=(10, 6))\n"
        "# organogram drawing here\n"
        "ax.set_aspect('equal')\n"
        "```"
    )


def build_stl_output_template():
    return (
        "Return exactly one JSON-compatible field:\n"
        "stl_code\n\n"
        "The code must be executable Python only and end with a valid `mesh = ...` assignment.\n"
        "Example:\n"
        "import trimesh\n"
        "import numpy as np\n"
        "mesh = trimesh.creation.box(extents=(40, 20, 10))"
    )


def split_summary_and_materials(text_no_code: str):
    """
    Split prose into conceptual summary and materials section.
    Supports headings like:
    - Materials
    - ## Materials
    - **Materials**
    - Bill of Materials / BOM
    """
    if not text_no_code:
        return "", None

    heading = re.search(
        r"(?im)^\s*(?:#+\s*)?(?:\*\*|__)?\s*(materials|bill of materials|bom)\s*(?:\*\*|__)?\s*:?\s*$",
        text_no_code
    )
    if heading:
        summary = text_no_code[:heading.start()].strip()
        materials = text_no_code[heading.start():].strip()
        return summary, materials if materials else None

    # Fallback for inline "Materials:" mentions
    inline = re.search(r"(?is)\b(materials|bill of materials|bom)\s*:\s*([\s\S]+)$", text_no_code)
    if inline:
        summary = text_no_code[:inline.start()].strip()
        materials = inline.group(0).strip()
        return summary, materials if materials else None

    return text_no_code.strip(), None


def fetch_ollama_models():
    """Return installed Ollama models and endpoint metadata."""
    base_url, configured_model, _ = get_ollama_config()
    headers = get_ollama_headers()
    last_status = None
    last_error = None

    for path in ('/api/tags', '/v1/models'):
        url = f"{base_url}{path}"
        try:
            r = requests.get(url, headers=headers, timeout=30)
        except requests.exceptions.RequestException as req_err:
            last_error = str(req_err)
            continue

        last_status = r.status_code
        if r.status_code == 200:
            payload = r.json()
            models = []

            # Native Ollama payload: {"models": [{"name": "..."}]}
            if isinstance(payload, dict) and isinstance(payload.get('models'), list):
                for model_item in payload.get('models', []):
                    if isinstance(model_item, dict):
                        name = model_item.get('name') or model_item.get('model')
                        if name:
                            models.append(str(name))
                    elif isinstance(model_item, str):
                        models.append(model_item)

            # OpenAI-style payload: {"data": [{"id": "..."}]}
            if not models and isinstance(payload, dict) and isinstance(payload.get('data'), list):
                for model_item in payload.get('data', []):
                    if isinstance(model_item, dict):
                        model_id = model_item.get('id') or model_item.get('name')
                        if model_id:
                            models.append(str(model_id))
                    elif isinstance(model_item, str):
                        models.append(model_item)

            return {
                'ok': True,
                'endpoint': url,
                'base_url': base_url,
                'configured_model': configured_model,
                'models': models,
                'model_count': len(models)
            }

        if r.status_code == 401:
            return {'ok': False, 'status': 401, 'error': 'Unauthorized', 'base_url': base_url}

    if last_error:
        return {'ok': False, 'status': 502, 'error': f'Unable to connect to Ollama at {base_url}: {last_error}', 'base_url': base_url}
    return {'ok': False, 'status': last_status or 500, 'error': 'Could not list Ollama models', 'base_url': base_url}


def generate_with_image_required(
    prompt_content: str,
    prompt_text: str,
    plot_instruction: str,
    max_attempts: int = 2,
    llm_timeout_sec: int = 90,
    llm_max_tokens: int = 1000,
    deadline_at: float = None,
    plot_output_template: str = "",
    request_id: str = ""
):
    """
    Run up to N LLM passes and require a valid matplotlib-rendered image.
    Returns: raw_response, summary_text, materials_text, plot_code, stl_code, image_base64, stl_bytes, llm_meta
    """
    def build_plot_attempt_prompt(base_instruction: str, last_error: str = "", attempt_idx: int = 1):
        if attempt_idx == 1:
            return base_instruction, 0.55

        correction_lines = [
            base_instruction,
            "",
            "Correction required:",
            f"- Previous attempt failed with: {last_error}",
            "- Return summary, materials, and one executable matplotlib code block only.",
            "- Do not return trimesh code in this call.",
            "- The python code must start with `import matplotlib.pyplot as plt`.",
            "- Prefer a simpler valid organogram over a complex invalid one.",
            "- Do not include placeholder text or recovery notes."
        ]
        if plot_output_template:
            correction_lines.extend([
                "",
                "Exact output template to follow:",
                plot_output_template
            ])
        return "\n".join(correction_lines), 0.2

    def build_stl_attempt_prompt(summary_text: str, materials_text: str, plot_code: str, last_error: str = "", attempt_idx: int = 1):
        lines = [
            f"Original prompt:\n{prompt_text}",
            "",
            "The matplotlib organogram is already accepted. Use it as the source of truth for geometry.",
            "",
            "Conceptual Summary:",
            summary_text or "(no summary)",
            "",
            "Materials:",
            materials_text or "(no materials)",
            "",
            "Accepted organogram code:",
            "```python",
            (plot_code or "").strip(),
            "```",
            "",
            "Now generate only the trimesh geometry code.",
            "Requirements for internal components (Mic/Speaker/Paths):",
            "- If a 'cube out of it' or 'hollow' is requested, use a larger box and subtract a slightly smaller one, or use `trimesh.creation.box(extents=(W,H,D))`. ",
            "- For 'mic and speaker inside', create small primitives (sphere/cone) and place them at internal coordinates.",
            "- Use `trimesh.util.concatenate` to group the outer shell and the inner components.",
            "- If the inner components should be visible, they must be part of the final mesh.",
            "- For 'paths', use `trimesh.creation.cylinder` or `capsule` connecting two points.",
            "",
            "General STL Requirements:",
            "- Return one JSON object only with field `stl_code`.",
            "- `stl_code` must contain executable Python only, with no markdown fences.",
            "- It must import trimesh.",
            "- It must assign the final geometry to `mesh`.",
            "- Add 3-6 comment lines with format: '# map: <organogram element> -> <geometry primitive> -> <acoustic function>'.",
            "- Keep geometry printable and non-empty."
        ]
        if attempt_idx > 1:
            lines.extend([
                "",
                "Correction required:",
                f"- Previous attempt failed with: {last_error}",
                "- Simplify the geometry if needed, but keep it explicitly tied to the organogram.",
                "- Do not omit `mesh = ...`."
            ])
        lines.extend([
            "",
            "Exact output template to follow:",
            build_stl_output_template()
        ])
        return "\n".join(lines), (0.3 if attempt_idx == 1 else 0.15)

    def compute_call_timeout(stage: str = "plot"):
        if deadline_at is not None:
            remaining = deadline_at - time.perf_counter()
            if remaining <= 8:
                if stage == "stl":
                    return None
                raise RuntimeError("Generation deadline exceeded before a valid organogram could be produced.")
        call_timeout = int(llm_timeout_sec) if int(llm_timeout_sec) > 0 else None
        if deadline_at is not None:
            # Keep margin for code execution/parsing after LLM response.
            remaining = max(1, int(deadline_at - time.perf_counter()))
            deadline_timeout = max(12, remaining - 5)
            if call_timeout is None:
                call_timeout = deadline_timeout
            else:
                call_timeout = max(12, min(call_timeout, deadline_timeout))
        return call_timeout

    def make_reasoning_callback(stage_label: str):
        if not request_id:
            return None
        def _callback(reasoning_preview: str):
            update_generation_progress(
                request_id,
                stage=stage_label,
                status='running',
                reasoning_preview=reasoning_preview
            )
        return _callback

    last_reason = "No valid matplotlib output produced."
    plot_schema = build_plot_generation_schema()
    summary_text = ""
    materials_text = None
    plot_code = ""
    image_base64 = None
    image_bytes = b""
    raw_plot_response = ""
    plot_llm_meta = None

    materials_prompt = get_specialized_prompt('materials')
    stl_prompt_content = get_specialized_prompt('digitalFab')

    # We try max_attempts. If we are on the last attempt and still failing, 
    # we switch to the fallback model to ensure we get a result.
    for attempt_idx in range(1, max_attempts + 1):
        is_last_attempt = (attempt_idx == max_attempts)
        current_model_override = None
        if is_last_attempt and max_attempts > 1:
            current_model_override = FALLBACK_OLLAMA_MODEL
            logging.warning(f"Using fallback model {FALLBACK_OLLAMA_MODEL} for the last attempt.")

        attempt_prompt, attempt_temperature = build_plot_attempt_prompt(
            plot_instruction,
            last_error=last_reason,
            attempt_idx=attempt_idx
        )
        call_timeout = compute_call_timeout(stage="plot")
        update_generation_progress(request_id, stage='plot_llm', status='running')

        try:
            structured_instruction = (
                attempt_prompt
                + "\n\nReturn one JSON object only. No markdown fences."
                + "\n- `summary`: conceptual summary text only."
                + "\n- `materials`: materials list as plain text lines."
                + "\n- `plot_code`: executable matplotlib code only, without fences."
            )
            structured_data, llm_meta = call_ollama_structured(
                materials_prompt,
                structured_instruction,
                schema=plot_schema,
                model_name_override=current_model_override,
                timeout=call_timeout,
                max_tokens=llm_max_tokens,
                temperature=min(attempt_temperature, 0.35),
                progress_callback=make_reasoning_callback('plot_llm')
            )
            summary_text = str(structured_data.get('summary') or '').strip()
            materials_text = str(structured_data.get('materials') or '').strip() or None
            plot_code = str(structured_data.get('plot_code') or '').strip()
            raw_plot_response = json.dumps(structured_data, ensure_ascii=False, indent=2)
            plot_llm_meta = llm_meta
            logging.info(
                f"LLM structured plot attempt={attempt_idx} endpoint={llm_meta.get('endpoint')} model={llm_meta.get('model')}"
            )
        except Exception as structured_err:
            logging.warning(f"Structured plot generation failed on attempt {attempt_idx}: {structured_err}")
            try:
                raw_plot_response, llm_meta = call_ollama_chat(
                    materials_prompt,
                    attempt_prompt,
                    model_name_override=current_model_override,
                    timeout=call_timeout,
                    max_tokens=llm_max_tokens,
                    temperature=attempt_temperature,
                    progress_callback=make_reasoning_callback('plot_llm')
                )
            except Exception as e:
                last_reason = f"LLM call failed: {e}"
                logging.error(last_reason)
                if 'requires more system memory' in str(e).lower():
                    raise RuntimeError(last_reason)
                continue
            plot_llm_meta = llm_meta
            logging.info(
                f"LLM freeform plot attempt={attempt_idx} endpoint={llm_meta.get('endpoint')} model={llm_meta.get('model')}"
            )

            text_no_code = re.sub(r"```[\s\S]*?```", "", raw_plot_response).strip()
            summary_text, materials_text = split_summary_and_materials(text_no_code)

            code_blocks = extract_python_code_blocks(raw_plot_response)
            plot_code, _ = split_plot_and_stl_code(code_blocks)

        if not plot_code:
            last_reason = "Model response did not include matplotlib code."
            continue

        try:
            update_generation_progress(request_id, stage='rendering_organogram', status='running')
            image_base64 = execute_matplotlib_code(plot_code)
            image_bytes = base64.b64decode(image_base64)
            if not image_bytes:
                last_reason = "Matplotlib code executed but produced empty image bytes."
                continue
        except Exception as e:
            last_reason = f"Matplotlib execution failed: {e}"
            logging.error(last_reason)
            continue
        break
    else:
        # Final fallback: if everything failed, try a very simple plot with fallback model
        logging.error("All plot attempts failed. Trying emergency fallback with simplest possible plot.")
        try:
            emergency_prompt = "Create a simplest possible Mantle Hood organogram for a musical instrument: one circle and one rectangle."
            structured_data, llm_meta = call_ollama_structured(
                prompt_content,
                emergency_prompt,
                schema=plot_schema,
                model_name_override=FALLBACK_OLLAMA_MODEL,
                timeout=90,
                max_tokens=512,
                temperature=0.1
            )
            summary_text = str(structured_data.get('summary') or 'Emergency fallback summary.').strip()
            plot_code = str(structured_data.get('plot_code') or '').strip()
            image_base64 = execute_matplotlib_code(plot_code)
            plot_llm_meta = llm_meta
        except Exception as final_e:
            raise RuntimeError(f"Complete generation failure including fallback: {last_reason} -> {final_e}")

    stl_schema = build_stl_generation_schema()
    stl_code = ""
    stl_bytes = None
    raw_stl_response = ""
    llm_meta = plot_llm_meta or {}
    stl_last_reason = "Model response did not include trimesh code block."

    for attempt_idx in range(1, max_attempts + 1):
        stl_attempt_prompt, stl_temperature = build_stl_attempt_prompt(
            summary_text=summary_text,
            materials_text=materials_text or "",
            plot_code=plot_code,
            last_error=stl_last_reason,
            attempt_idx=attempt_idx
        )
        call_timeout = compute_call_timeout(stage="stl")
        if call_timeout is None:
            logging.warning("Skipping LLM STL generation because request deadline is nearly exhausted.")
            break
        update_generation_progress(request_id, stage='stl_llm', status='running')

        try:
            structured_data, stl_meta = call_ollama_structured(
                stl_prompt_content,
                stl_attempt_prompt,
                schema=stl_schema,
                timeout=call_timeout,
                max_tokens=max(256, min(llm_max_tokens, 900)),
                temperature=stl_temperature,
                progress_callback=make_reasoning_callback('stl_llm')
            )
            stl_code = str(structured_data.get('stl_code') or '').strip()
            raw_stl_response = json.dumps(structured_data, ensure_ascii=False, indent=2)
            llm_meta = stl_meta or llm_meta
            logging.info(
                f"LLM structured stl attempt={attempt_idx} endpoint={llm_meta.get('endpoint')} model={llm_meta.get('model')}"
            )
        except Exception as structured_err:
            logging.warning(f"Structured STL generation failed on attempt {attempt_idx}: {structured_err}")
            try:
                raw_stl_response, stl_meta = call_ollama_chat(
                    stl_prompt_content,
                    stl_attempt_prompt,
                    timeout=call_timeout,
                    max_tokens=max(256, min(llm_max_tokens, 900)),
                    temperature=stl_temperature,
                    progress_callback=make_reasoning_callback('stl_llm')
                )
            except Exception as e:
                stl_last_reason = f"LLM call failed: {e}"
                logging.error(stl_last_reason)
                if 'requires more system memory' in str(e).lower():
                    break
                continue
            llm_meta = stl_meta or llm_meta
            logging.info(
                f"LLM freeform stl attempt={attempt_idx} endpoint={llm_meta.get('endpoint')} model={llm_meta.get('model')}"
            )
            stl_blocks = extract_python_code_blocks(raw_stl_response)
            _, stl_code = split_plot_and_stl_code(stl_blocks)

        if not stl_code:
            stl_last_reason = "Model response did not include trimesh code block."
            continue

        try:
            update_generation_progress(request_id, stage='building_geometry', status='running')
            stl_bytes = execute_trimesh_code(stl_code)
            break
        except Exception as e:
            stl_last_reason = f"Trimesh execution failed: {e}"
            logging.error(stl_last_reason)
            stl_bytes = None
            continue

    if stl_bytes is None and stl_last_reason:
        logging.warning(f"STL generation will use fallback if available. Last reason: {stl_last_reason}")

    raw_response = "\n\n".join(part for part in [raw_plot_response, raw_stl_response] if part).strip()
    return (
        raw_response,
        summary_text,
        materials_text,
        plot_code,
        stl_code,
        image_base64,
        stl_bytes,
        llm_meta
    )

def execute_matplotlib_code(code: str) -> str:
    """Execute matplotlib code safely and return the PNG as base64.
    - Strips code fences
    - Removes plt.show()
    - Ensures trailing newline
    - Balances common brackets if SyntaxError occurs
    """
    # Normalize and sanitize code first
    cleaned = code.strip()
    # Strip stray code fences if present
    cleaned = re.sub(r'^```(?:python|py)?', '', cleaned, flags=re.IGNORECASE | re.MULTILINE)
    cleaned = re.sub(r'```\s*$', '', cleaned, flags=re.MULTILINE)
    # Remove plt.show() to avoid blocking
    cleaned = re.sub(r'plt\.show\(\s*\)', '', cleaned)
    cleaned = _repair_matplotlib_imports(cleaned)
    # Ensure trailing newline
    if not cleaned.endswith('\n'):
        cleaned += '\n'

    def balance_brackets(s: str) -> str:
        pairs = [('(', ')'), ('[', ']'), ('{', '}')]
        for open_c, close_c in pairs:
            missing = s.count(open_c) - s.count(close_c)
            if missing > 0:
                s += close_c * missing + '\n'
        return s

    def looks_like_narrative_line(line: str) -> bool:
        stripped = (line or '').strip()
        if not stripped:
            return False
        if re.match(r'^\d+\.\s+[A-Za-z].*$', stripped):
            return True
        if re.match(r'^[-*]\s+[A-Za-z].*$', stripped):
            return True
        if re.match(r'^[A-Z][A-Z0-9 _-]{3,}$', stripped):
            return True
        if stripped.lower().startswith(('conceptual summary', 'materials:', 'summary:')):
            return True
        return False

    code_to_run = cleaned
    # Try pre-compilation; recover from common narrative lines before fallback repair.
    try:
        compile(code_to_run, '<string>', 'exec')
    except SyntaxError as syn_err:
        lines = cleaned.splitlines()
        recovered = False

        # Remove obvious non-Python narrative lines that models sometimes inject.
        for _ in range(8):
            lineno = getattr(syn_err, 'lineno', None)
            if not lineno or lineno < 1 or lineno > len(lines):
                break
            bad_line = lines[lineno - 1]
            if not looks_like_narrative_line(bad_line):
                break
            lines.pop(lineno - 1)
            candidate = "\n".join(lines).strip()
            if candidate:
                candidate += "\n"
            try:
                compile(candidate, '<string>', 'exec')
                code_to_run = candidate
                recovered = True
                break
            except SyntaxError as next_err:
                syn_err = next_err
                continue

        if not recovered:
            code_to_run = balance_brackets(cleaned)
            code_to_run = _compile_with_truncation(code_to_run, '<string>')

    # Execute and capture the current figure
    try:
        plt.close('all')

        # LLM-generated code often references patch classes through plt.<Class> or without imports.
        # These aliases/locals improve execution robustness without mutating the generated logic.
        def _safe_arrow(x, y, dx, dy, *args, **kwargs):
            color = kwargs.pop('color', kwargs.pop('fc', kwargs.pop('facecolor', 'white')))
            linewidth = kwargs.pop('linewidth', kwargs.pop('lw', 1.5))
            alpha = kwargs.pop('alpha', 1.0)
            arrowstyle = kwargs.pop('arrowstyle', '-|>')
            head_width = kwargs.pop('head_width', 0.25)
            mutation_scale = kwargs.pop('mutation_scale', max(8.0, float(head_width) * 20.0))
            return FancyArrowPatch(
                (x, y),
                (x + dx, y + dy),
                arrowstyle=arrowstyle,
                mutation_scale=mutation_scale,
                color=color,
                linewidth=linewidth,
                alpha=alpha
            )

        def _safe_fancy_arrow_patch(*args, **kwargs):
            # Accept both FancyArrowPatch(posA, posB, ...) and Arrow-like (x, y, dx, dy, ...) signatures.
            # Some model outputs mix annotate-style kwargs that FancyArrowPatch does not accept.
            xy = kwargs.pop('xy', None)
            xytext = kwargs.pop('xytext', None)
            arrowprops = kwargs.pop('arrowprops', None)
            if xytext is not None and 'posA' not in kwargs and len(args) < 1:
                kwargs['posA'] = xytext
            if xy is not None and 'posB' not in kwargs and len(args) < 2:
                kwargs['posB'] = xy
            if isinstance(arrowprops, dict):
                # Accept annotate-style arrowprops by mapping only known FancyArrowPatch kwargs.
                for key in (
                    'arrowstyle', 'connectionstyle', 'shrinkA', 'shrinkB',
                    'mutation_scale', 'mutation_aspect', 'alpha', 'zorder',
                    'linewidth', 'lw', 'linestyle', 'ls',
                    'color', 'edgecolor', 'ec', 'facecolor', 'fc',
                    'patchA', 'patchB', 'capstyle', 'joinstyle'
                ):
                    if key in arrowprops and key not in kwargs:
                        kwargs[key] = arrowprops[key]
            kwargs.pop('xycoords', None)
            kwargs.pop('textcoords', None)
            kwargs.pop('annotation_clip', None)
            if len(args) >= 4 and all(isinstance(v, (int, float)) for v in args[:4]):
                return _safe_arrow(*args, **kwargs)
            kwargs.pop('head_width', None)
            kwargs.pop('head_length', None)
            kwargs.pop('width', None)
            return FancyArrowPatch(*args, **kwargs)

        if not hasattr(Polygon, 'copy'):
            def _polygon_copy(self):
                return Polygon(
                    self.get_xy(),
                    closed=getattr(self, '_closed', True),
                    facecolor=self.get_facecolor(),
                    edgecolor=self.get_edgecolor(),
                    linewidth=self.get_linewidth(),
                    alpha=self.get_alpha()
                )
            try:
                Polygon.copy = _polygon_copy
            except Exception:
                pass

        def _normalize_font_variant(value):
            if isinstance(value, (list, tuple, set)):
                if not value:
                    return 'normal'
                value = next(iter(value))
            if value is None:
                return 'normal'
            text = str(value).strip().lower()
            if text in ('normal', 'small-caps'):
                return text
            return 'normal'

        try:
            from matplotlib.font_manager import FontProperties as _FontProperties
            from matplotlib.text import Text as _Text

            _orig_set_variant = getattr(_FontProperties, 'set_variant', None)
            if callable(_orig_set_variant) and not getattr(_orig_set_variant, '_soog_safe', False):
                def _safe_set_variant(self, variant):
                    return _orig_set_variant(self, _normalize_font_variant(variant))
                _safe_set_variant._soog_safe = True
                _FontProperties.set_variant = _safe_set_variant

            _orig_text_set_fontvariant = getattr(_Text, 'set_fontvariant', None)
            if callable(_orig_text_set_fontvariant) and not getattr(_orig_text_set_fontvariant, '_soog_safe', False):
                def _safe_text_set_fontvariant(self, variant):
                    return _orig_text_set_fontvariant(self, _normalize_font_variant(variant))
                _safe_text_set_fontvariant._soog_safe = True
                _Text.set_fontvariant = _safe_text_set_fontvariant
        except Exception:
            pass

        setattr(plt, 'Arc', Arc)
        setattr(plt, 'Ellipse', Ellipse)
        setattr(plt, 'Polygon', Polygon)
        setattr(plt, 'Circle', Circle)
        setattr(plt, 'Rectangle', Rectangle)
        setattr(plt, 'RegularPolygon', RegularPolygon)
        setattr(plt, 'Arrow', _safe_arrow)
        setattr(plt, 'FancyArrowPatch', _safe_fancy_arrow_patch)
        try:
            import matplotlib.patches as _mpatches
            setattr(_mpatches, 'FancyArrowPatch', _safe_fancy_arrow_patch)
        except Exception:
            pass

        safe_globals = {
            '__builtins__': __builtins__,
            'matplotlib': matplotlib,
            'np': np,
            'plt': plt,
            'Arc': Arc,
            'Circle': Circle,
            'Ellipse': Ellipse,
            'FancyArrowPatch': _safe_fancy_arrow_patch,
            'Line2D': Line2D,
            'Polygon': Polygon,
            'Rectangle': Rectangle,
            'RegularPolygon': RegularPolygon,
        }
        exec(code_to_run, safe_globals, safe_globals)

        fig = plt.gcf()
        if not fig.axes:
            raise RuntimeError("Matplotlib code executed but produced no axes")
        has_visible_content = any(
            ax.has_data() or ax.patches or ax.lines or ax.collections or ax.images or ax.texts
            for ax in fig.axes
        )
        if not has_visible_content:
            raise RuntimeError("Matplotlib code executed but produced an empty figure")

        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode('utf-8')
    except Exception as e:
        logging.error(f"Error executing matplotlib code: {e}")
        raise


def _compile_with_truncation(code: str, filename: str):
    """Compile code; if trailing lines are broken, truncate until a valid prefix is found."""
    try:
        compile(code, filename, 'exec')
        return code
    except SyntaxError:
        lines = code.splitlines()
        # Keep at least a few lines; drop from the end to recover from partial model output.
        for keep in range(len(lines) - 1, 3, -1):
            candidate = "\n".join(lines[:keep]).strip()
            if not candidate:
                continue
            candidate += "\n"
            try:
                compile(candidate, filename, 'exec')
                return candidate
            except SyntaxError:
                continue
        raise


def _stable_unit(seed_text: str, salt: str) -> float:
    digest = hashlib.sha256(f"{salt}|{seed_text}".encode('utf-8', errors='ignore')).hexdigest()
    return int(digest[:8], 16) / float(0xFFFFFFFF)


def _infer_stl_profile(prompt: str = "", summary_text: str = "", materials_text: str = "", plot_code: str = "") -> str:
    text = " ".join([prompt or "", summary_text or "", materials_text or "", plot_code or ""]).lower()
    profile_keywords = {
        'aerophone': ['aerophone', 'flute', 'pipe', 'reed', 'organ', 'wind', 'air column', 'mouthpiece'],
        'chordophone': ['chordophone', 'string', 'guitar', 'violin', 'harp', 'lute', 'bowed', 'plucked'],
        'membranophone': ['membranophone', 'drum', 'membrane', 'skin', 'percussion'],
        'idiophone': ['idiophone', 'plate', 'bell', 'gong', 'bar', 'chime', 'xylophone'],
        'electro': ['electro', 'speaker', 'microphone', 'sensor', 'dmi', 'hybrid', 'digital', 'interface']
    }
    scores = {name: 0 for name in profile_keywords}
    for name, words in profile_keywords.items():
        for w in words:
            if w in text:
                scores[name] += 1

    # Extra cues from organogram code.
    lowered_plot = (plot_code or "").lower()
    if 'circle(' in lowered_plot:
        scores['aerophone'] += 1
    if 'rectangle(' in lowered_plot:
        scores['chordophone'] += 1
    if 'polygon(' in lowered_plot:
        scores['idiophone'] += 1

    winner = max(scores, key=scores.get)
    return winner if scores[winner] > 0 else 'hybrid'


def _build_contextual_fallback_stl_code(prompt: str = "", summary_text: str = "", materials_text: str = "", plot_code: str = ""):
    seed_text = " ".join([prompt or "", summary_text or "", materials_text or "", plot_code or ""]).strip() or "soog"
    profile = _infer_stl_profile(prompt, summary_text, materials_text, plot_code)

    u1 = _stable_unit(seed_text, 'u1')
    u2 = _stable_unit(seed_text, 'u2')
    u3 = _stable_unit(seed_text, 'u3')
    u4 = _stable_unit(seed_text, 'u4')

    sections = 28 + int(u1 * 36)  # 28..64
    core_length = round(130 + u2 * 170, 2)  # 130..300
    core_radius = round(8 + u3 * 16, 2)  # 8..24
    aux_scale = round(0.35 + u4 * 0.55, 3)  # 0.35..0.9

    if profile == 'aerophone':
        bell_height = round(core_length * (0.18 + aux_scale * 0.15), 2)
        bell_radius = round(core_radius * (1.15 + aux_scale * 0.6), 2)
        mouth_height = round(core_length * (0.12 + aux_scale * 0.08), 2)
        mouth_radius = round(core_radius * (0.42 + aux_scale * 0.18), 2)
        return f"""import numpy as np
import trimesh

# map: air-column -> cylinder, bell radiation -> cone, mouthpiece interface -> cylinder
body = trimesh.creation.cylinder(radius={core_radius}, height={core_length}, sections={sections})
bell = trimesh.creation.cone(radius={bell_radius}, height={bell_height}, sections={sections})
bell.apply_translation((0, 0, -{round(core_length / 2 + bell_height / 2, 2)}))
mouth = trimesh.creation.cylinder(radius={mouth_radius}, height={mouth_height}, sections={sections})
mouth.apply_translation((0, 0, {round(core_length / 2 + mouth_height / 2, 2)}))
spine = trimesh.creation.box(extents=({round(core_radius * 0.8, 2)}, {round(core_radius * 1.1, 2)}, {round(core_length * 0.4, 2)}))
mesh = trimesh.util.concatenate([body, bell, mouth, spine])
""", profile

    if profile == 'chordophone':
        body_x = round(core_length * 0.9, 2)
        body_y = round(core_radius * 5.5, 2)
        body_z = round(core_radius * 1.35, 2)
        neck_x = round(core_length * 0.75, 2)
        neck_y = round(core_radius * 1.25, 2)
        neck_z = round(core_radius * 0.85, 2)
        return f"""import numpy as np
import trimesh

# map: resonator plate -> box, neck interface -> box, bridge coupling -> bar
resonator = trimesh.creation.box(extents=({body_x}, {body_y}, {body_z}))
neck = trimesh.creation.box(extents=({neck_x}, {neck_y}, {neck_z}))
neck.apply_translation(({round(body_x * 0.62, 2)}, 0, {round(body_z * 0.45, 2)}))
bridge = trimesh.creation.box(extents=({round(body_x * 0.18, 2)}, {round(body_y * 0.16, 2)}, {round(body_z * 0.42, 2)}))
bridge.apply_translation(({round(-body_x * 0.1, 2)}, 0, {round(body_z * 0.72, 2)}))
peg = trimesh.creation.cylinder(radius={round(core_radius * 0.22, 2)}, height={round(body_y * 0.75, 2)}, sections={sections})
peg.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
peg.apply_translation(({round(body_x * 0.94, 2)}, 0, {round(body_z * 0.6, 2)}))
mesh = trimesh.util.concatenate([resonator, neck, bridge, peg])
""", profile

    if profile == 'membranophone':
        shell_h = round(core_length * 0.55, 2)
        shell_r = round(core_radius * 1.35, 2)
        ring_h = round(max(2.0, shell_h * 0.045), 2)
        return f"""import numpy as np
import trimesh

# map: shell cavity -> cylinder, membrane rims -> rings, support feet -> boxes
shell = trimesh.creation.cylinder(radius={shell_r}, height={shell_h}, sections={sections})
rim_top = trimesh.creation.cylinder(radius={round(shell_r * 1.06, 2)}, height={ring_h}, sections={sections})
rim_top.apply_translation((0, 0, {round(shell_h * 0.5, 2)}))
rim_bottom = trimesh.creation.cylinder(radius={round(shell_r * 1.06, 2)}, height={ring_h}, sections={sections})
rim_bottom.apply_translation((0, 0, {round(-shell_h * 0.5, 2)}))
foot_a = trimesh.creation.box(extents=({round(shell_r * 0.55, 2)}, {round(shell_r * 0.2, 2)}, {round(shell_h * 0.25, 2)}))
foot_a.apply_translation(({round(shell_r * 0.65, 2)}, 0, {round(-shell_h * 0.45, 2)}))
foot_b = foot_a.copy()
foot_b.apply_translation(({round(-shell_r * 1.3, 2)}, 0, 0))
mesh = trimesh.util.concatenate([shell, rim_top, rim_bottom, foot_a, foot_b])
""", profile

    if profile == 'idiophone':
        plate_x = round(core_length * 0.95, 2)
        plate_y = round(core_radius * 3.8, 2)
        plate_z = round(core_radius * 0.55, 2)
        return f"""import numpy as np
import trimesh

# map: striking plate -> box, resonant bars -> cylinders, frame -> rails
plate = trimesh.creation.box(extents=({plate_x}, {plate_y}, {plate_z}))
bar_a = trimesh.creation.cylinder(radius={round(core_radius * 0.18, 2)}, height={round(plate_x * 0.9, 2)}, sections={sections})
bar_a.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
bar_a.apply_translation((0, {round(plate_y * 0.32, 2)}, {round(plate_z * 1.15, 2)}))
bar_b = bar_a.copy()
bar_b.apply_translation((0, {round(-plate_y * 0.64, 2)}, 0))
rail_l = trimesh.creation.box(extents=({round(plate_x, 2)}, {round(plate_y * 0.08, 2)}, {round(plate_z * 1.5, 2)}))
rail_l.apply_translation((0, {round(plate_y * 0.52, 2)}, 0))
rail_r = rail_l.copy()
rail_r.apply_translation((0, {round(-plate_y * 1.04, 2)}, 0))
mesh = trimesh.util.concatenate([plate, bar_a, bar_b, rail_l, rail_r])
""", profile

    if profile == 'electro':
        body_x = round(core_length * 0.65, 2)
        body_y = round(core_radius * 4.8, 2)
        body_z = round(core_radius * 2.1, 2)
        horn_h = round(core_radius * (2.4 + aux_scale * 1.8), 2)
        horn_r = round(core_radius * (0.7 + aux_scale * 0.45), 2)
        return f"""import numpy as np
import trimesh

# map: electronic core -> chassis box, transducers -> conical emitters, control node -> sphere
chassis = trimesh.creation.box(extents=({body_x}, {body_y}, {body_z}))
horn_l = trimesh.creation.cone(radius={horn_r}, height={horn_h}, sections={sections})
horn_l.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
horn_l.apply_translation(({round(body_x * 0.54, 2)}, {round(body_y * 0.28, 2)}, 0))
horn_r = horn_l.copy()
horn_r.apply_translation((0, {round(-body_y * 0.56, 2)}, 0))
control = trimesh.creation.icosphere(subdivisions=2, radius={round(core_radius * 0.48, 2)})
control.apply_translation(({round(-body_x * 0.28, 2)}, 0, {round(body_z * 0.72, 2)}))
mesh = trimesh.util.concatenate([chassis, horn_l, horn_r, control])
""", profile

    # Hybrid fallback profile.
    return f"""import numpy as np
import trimesh

# map: core resonance -> cylinder, interaction body -> box, nodal focus -> sphere
core = trimesh.creation.cylinder(radius={core_radius}, height={round(core_length * 0.72, 2)}, sections={sections})
body = trimesh.creation.box(extents=({round(core_length * 0.55, 2)}, {round(core_radius * 3.2, 2)}, {round(core_radius * 1.6, 2)}))
body.apply_translation((0, 0, {round(core_radius * 0.52, 2)}))
node = trimesh.creation.icosphere(subdivisions=2, radius={round(core_radius * 0.5, 2)})
node.apply_translation(({round(core_radius * 0.1, 2)}, {round(core_radius * 0.1, 2)}, {round(core_length * 0.42, 2)}))
mesh = trimesh.util.concatenate([core, body, node])
""", profile


def _build_legacy_fallback_stl_bytes() -> bytes:
    if trimesh is None:
        raise RuntimeError("trimesh is not installed")
    body = trimesh.creation.cylinder(radius=18, height=180, sections=48)
    mouth = trimesh.creation.cylinder(radius=10, height=35, sections=32)
    mouth.apply_translation((0, 0, 107.5))
    base = trimesh.creation.box(extents=(24, 24, 10))
    base.apply_translation((0, 0, -95))
    mesh = trimesh.util.concatenate([body, mouth, base])
    data = mesh.export(file_type='stl')
    return data if isinstance(data, (bytes, bytearray)) else str(data).encode('utf-8')


def build_fallback_stl_bytes(
    prompt: str = "",
    summary_text: str = "",
    materials_text: str = "",
    plot_code: str = ""
):
    """
    Build deterministic but prompt-sensitive STL fallback.
    Returns: (stl_bytes, stl_code, profile)
    """
    stl_code, profile = _build_contextual_fallback_stl_code(prompt, summary_text, materials_text, plot_code)
    try:
        stl_bytes = execute_trimesh_code(stl_code)
        return stl_bytes, stl_code, profile
    except Exception as e:
        logging.error(f"Contextual STL fallback failed ({profile}): {e}")
        stl_bytes = _build_legacy_fallback_stl_bytes()
        legacy_code = (
            "import trimesh\n"
            "# legacy fallback profile used after contextual fallback failure\n"
            "body = trimesh.creation.cylinder(radius=18, height=180, sections=48)\n"
            "mouth = trimesh.creation.cylinder(radius=10, height=35, sections=32)\n"
            "mouth.apply_translation((0, 0, 107.5))\n"
            "base = trimesh.creation.box(extents=(24, 24, 10))\n"
            "base.apply_translation((0, 0, -95))\n"
            "mesh = trimesh.util.concatenate([body, mouth, base])\n"
        )
        return stl_bytes, legacy_code, "legacy"


def execute_trimesh_code(code: str) -> bytes:
    """Execute Python code that builds a trimesh Trimesh/Scene and return STL bytes."""
    if trimesh is None:
        raise RuntimeError("trimesh is not installed; install to enable STL mode")

    cleaned = code.strip()
    cleaned = re.sub(r'^```(?:python|py)?', '', cleaned, flags=re.IGNORECASE | re.MULTILINE)
    cleaned = re.sub(r'```\s*$', '', cleaned, flags=re.MULTILINE)
    if ShapelyPolygon is None:
        cleaned = re.sub(r'^\s*(?:from|import)\s+shapely[^\n]*\n?', '', cleaned, flags=re.MULTILINE)
        if re.search(r'\b(LineString|Point|Polygon|unary_union)\b', cleaned):
            raise RuntimeError("shapely is not installed; generate trimesh code using pure trimesh primitives")
    if not cleaned.endswith('\n'):
        cleaned += '\n'
    cleaned = _compile_with_truncation(cleaned, '<stl>')

    # Compat helper: some models call creation.extrude_prism although many trimesh builds expose extrude_polygon.
    def _extrude_prism_compat(polygon, height):
        return trimesh.creation.extrude_polygon(polygon, height)

    # Constrained globals for execution
    g = {
        '__builtins__': __builtins__,
        'trimesh': trimesh,
        'np': np,
    }
    if ShapelyPolygon is not None:
        g['Polygon'] = ShapelyPolygon
    if ShapelyPoint is not None:
        g['Point'] = ShapelyPoint
    if ShapelyLineString is not None:
        g['LineString'] = ShapelyLineString
    if shapely_unary_union is not None:
        g['unary_union'] = shapely_unary_union
    if not hasattr(trimesh.creation, 'extrude_prism') and hasattr(trimesh.creation, 'extrude_polygon'):
        g['extrude_prism'] = _extrude_prism_compat

    l = {}
    exec(cleaned, g, l)

    mesh = l.get('mesh') or g.get('mesh')
    scene = l.get('scene') or g.get('scene')
    if mesh is None and scene is None:
        # try to find any Trimesh instance
        for v in list(l.values()) + list(g.values()):
            if hasattr(trimesh, 'Trimesh') and isinstance(v, trimesh.Trimesh):
                mesh = v
                break
        if mesh is None and hasattr(trimesh, 'Scene'):
            for v in list(l.values()) + list(g.values()):
                if isinstance(v, trimesh.Scene):
                    scene = v
                    break

    if scene is not None and hasattr(scene, 'dump'):
        geoms = []
        for _, geom in scene.geometry.items():
            if isinstance(geom, trimesh.Trimesh):
                geoms.append(geom)
        if geoms:
            mesh = trimesh.util.concatenate(geoms)

    if mesh is None:
        if scene is not None and hasattr(scene, 'export'):
            data = scene.export(file_type='stl')
            if isinstance(data, (bytes, bytearray)) and len(data) > 84:
                return data
            if isinstance(data, str) and len(data.strip()) > 0:
                return data.encode('utf-8')
        raise RuntimeError("No `mesh` or exportable `scene` found after executing trimesh code")

    face_count = int(getattr(mesh, 'faces', np.empty((0, 3))).shape[0]) if hasattr(mesh, 'faces') else 0
    if face_count <= 0:
        raise RuntimeError("Generated mesh has zero faces")

    data = mesh.export(file_type='stl')
    if not data:
        raise RuntimeError("Mesh export produced empty STL output")
    return data if isinstance(data, (bytes, bytearray)) else str(data).encode('utf-8')

def _extract_first_line(text: str) -> str:
    for line in (text or "").splitlines():
        cleaned = line.strip().lstrip('#').strip()
        if cleaned:
            return cleaned
    return ""


def _descriptive_slug(primary_text: str, fallback_text: str = "") -> str:
    """
    Build a descriptive title slug using one adjective + at least two nouns.
    Priority source is the first line of primary_text (generation output).
    """
    stop = {
        'the', 'and', 'for', 'with', 'that', 'this', 'from', 'into', 'over', 'under', 'using',
        'your', 'a', 'an', 'of', 'to', 'in', 'on', 'by', 'or', 'as', 'is', 'are', 'be', 'it',
        'its', 'their', 'our', 'you', 'then', 'first', 'finally', 'section', 'materials',
        'conceptual', 'summary'
    }
    adjective_bank = {
        'acoustic', 'resonant', 'hybrid', 'modular', 'speculative', 'dynamic', 'adaptive',
        'harmonic', 'industrial', 'organological', 'performative', 'digital', 'analog',
        'virtual', 'structural', 'aerodynamic', 'geometric', 'mechanical'
    }

    first_line = _extract_first_line(primary_text) or _extract_first_line(fallback_text)
    source = f"{first_line} {fallback_text or ''}".strip()
    words = [w for w in re.findall(r"[a-zA-Z]{3,}", source.lower()) if w not in stop]

    adjective = None
    nouns = []
    for w in words:
        if adjective is None and (
            w in adjective_bank or w.endswith(('ive', 'ous', 'al', 'ic', 'ary', 'ory', 'ful', 'less', 'ant', 'ent'))
        ):
            adjective = w
            continue
        nouns.append(w)

    # Ensure we get at least two nouns
    unique_nouns = []
    for noun in nouns:
        if noun not in unique_nouns:
            unique_nouns.append(noun)
        if len(unique_nouns) >= 2:
            break

    if len(unique_nouns) < 2:
        for w in words:
            if w not in unique_nouns and w != adjective:
                unique_nouns.append(w)
            if len(unique_nouns) >= 2:
                break

    if adjective is None:
        adjective = 'hybrid'
    while len(unique_nouns) < 2:
        unique_nouns.append('organogram')

    parts = [adjective, unique_nouns[0], unique_nouns[1]]
    return "_".join(parts)


def _save_gallery_item(
    prompt: str,
    answer: str,
    code: str,
    image_bytes: bytes,
    stl_bytes: bytes = None,
    sketch_bytes: bytes = None,
    llm_model: str = None,
    elapsed_ms: int = None,
    plot_code: str = None,
    stl_code: str = None,
    materials_text: str = None,
    summary_text: str = None,
    sketch_prompt: str = None,
    sketch_model: str = None,
    refact_meta: dict = None
) -> dict:
    refact_meta = refact_meta or {}
    is_refact = bool(refact_meta.get('is_refact'))
    source_basename = (refact_meta.get('source') or refact_meta.get('base') or '').strip()

    ts = datetime.datetime.utcnow().strftime('%Y%m%d-%H%M%S')
    requested_title = (refact_meta.get('title') or '').strip()
    descriptor = requested_title or _descriptive_slug(summary_text or answer, fallback_text=prompt)
    title_slug = _safe_slug(descriptor, fallback='hybrid_organogram_system')
    title = title_slug.replace('_', '-')

    if is_refact:
        group_candidate = _safe_slug(
            (refact_meta.get('group') or refact_meta.get('group_id') or '').strip(),
            fallback=''
        )
        group_id = group_candidate or _safe_slug(source_basename or title_slug, fallback=f"{title_slug}_group")
        defaults = _group_defaults(group_id)
        if defaults.get('title_slug'):
            title_slug = _safe_slug(defaults.get('title_slug'), fallback=title_slug)
            title = (defaults.get('title') or title).strip() or title
        seed_version_index = _version_index_from_value(
            refact_meta.get('version_index') or refact_meta.get('version'),
            default=1
        )
        version_index = _next_group_version_index(group_id, seed_version_index=seed_version_index)
    else:
        group_id = f"{title_slug}_{ts}"
        version_index = 1

    version = _version_label_from_index(version_index)
    version_token = version
    base = f"{ts}_{title_slug}_{version_token}"
    png_path = os.path.join(GALLERY_DIR, f"{base}.png")
    txt_path = os.path.join(GALLERY_DIR, f"{base}.txt")
    json_path = os.path.join(GALLERY_DIR, f"{base}.json")
    stl_path = os.path.join(GALLERY_DIR, f"{base}.stl") if stl_bytes else None
    sketch_path = os.path.join(GALLERY_DIR, f"{base}.sketch.png") if sketch_bytes else None

    # Resolve collisions (rare but possible with concurrent writes on same second).
    suffix = 1
    while os.path.exists(json_path):
        base = f"{ts}_{title_slug}_{version_token}_{suffix}"
        png_path = os.path.join(GALLERY_DIR, f"{base}.png")
        txt_path = os.path.join(GALLERY_DIR, f"{base}.txt")
        json_path = os.path.join(GALLERY_DIR, f"{base}.json")
        stl_path = os.path.join(GALLERY_DIR, f"{base}.stl") if stl_bytes else None
        sketch_path = os.path.join(GALLERY_DIR, f"{base}.sketch.png") if sketch_bytes else None
        suffix += 1

    plot_code = (plot_code or code or '').strip()
    stl_code = (stl_code or '').strip()
    summary_stored = (summary_text or answer or '').strip()
    materials_stored = (materials_text or '').strip() or None
    combined_code = plot_code or stl_code or ''

    # write image only if provided
    has_image = bool(image_bytes) and len(image_bytes) > 0
    if has_image:
        try:
            with open(png_path, 'wb') as f:
                f.write(image_bytes)
        except Exception as e:
            logging.error(f"Error writing PNG file: {e}")
            has_image = False

    # write text
    try:
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("# Prompt\n\n")
            f.write(prompt.strip() + "\n\n")
            if summary_stored:
                f.write("# Summary\n\n")
                f.write(summary_stored + "\n")
            if materials_stored:
                f.write("\n# Materials\n\n")
                f.write(materials_stored + "\n")
            if llm_model or elapsed_ms is not None:
                f.write("\n# Generation Metadata\n\n")
                if llm_model:
                    f.write(f"- model: {llm_model}\n")
                if elapsed_ms is not None:
                    f.write(f"- elapsed_ms: {int(elapsed_ms)}\n")
                if sketch_model:
                    f.write(f"- sketch_model: {sketch_model}\n")
                f.write(f"- group_id: {group_id}\n")
                f.write(f"- version: {version}\n")
                if source_basename:
                    f.write(f"- source: {source_basename}\n")
            if plot_code:
                f.write("\n# Plot Code (matplotlib)\n\n")
                f.write(plot_code + "\n")
            if stl_code:
                f.write("\n# Geometry Code (trimesh)\n\n")
                f.write(stl_code + "\n")
            if sketch_prompt:
                f.write("\n# Sketch Prompt\n\n")
                f.write(sketch_prompt.strip() + "\n")
    except Exception:
        pass

    # write STL if provided
    if stl_bytes:
        try:
            with open(stl_path, 'wb') as f:
                f.write(stl_bytes)
        except Exception as e:
            logging.error(f"Error writing STL file: {e}")
            stl_path = None

    has_sketch = bool(sketch_bytes) and len(sketch_bytes) > 0
    if has_sketch:
        try:
            with open(sketch_path, 'wb') as f:
                f.write(sketch_bytes)
        except Exception as e:
            logging.error(f"Error writing sketch PNG file: {e}")
            has_sketch = False

    meta = {
        'basename': base,
        'timestamp': ts,
        'title': title,
        'title_slug': title_slug,
        'group_id': group_id,
        'version': version,
        'version_index': version_index,
        'source_basename': source_basename or None,
        'prompt': prompt,
        'answer': answer,
        'summary': summary_stored or None,
        'materials_text': materials_stored,
        'code': combined_code,
        'plot_code': plot_code,
        'stl_code': stl_code,
        'sketch_prompt': (sketch_prompt or '').strip() or None,
        'sketch_model': (sketch_model or '').strip() or None,
        'llm_model': llm_model,
        'elapsed_ms': int(elapsed_ms) if elapsed_ms is not None else None,
        'image_url': f"/api/gallery/image/{base}.png" if has_image else None,
        'stl_url': f"/api/gallery/file/{base}.stl" if stl_bytes and stl_path else None,
        'sketch_url': f"/api/gallery/image/{base}.sketch.png" if has_sketch else None,
        'modes': [
            m for m in
            (['plot'] if has_image else [])
            + (['stl'] if stl_bytes and stl_path else [])
            + (['sketch'] if has_sketch else [])
        ]
    }
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f)
    return meta

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


@app.route('/api/generate/progress/<string:request_id>', methods=['GET'])
def generate_progress(request_id):
    current = get_generation_progress(request_id)
    if not current:
        return jsonify({'ok': False, 'status': 'missing'}), 404
    return jsonify({'ok': True, **current})

@app.route('/api/generate', methods=['POST'])
@log_activity('generate')
def generate():
    data = request.json
    request_id = str((data or {}).get('request_id') or '').strip()
    prompt_input = data.get('prompt', '').strip()
    prompt, refact_meta = parse_refact_marker(prompt_input)
    if not prompt and prompt_input:
        prompt = prompt_input
    if not prompt:
        return jsonify({'error': 'No prompt provided.'}), 400
    update_generation_progress(
        request_id,
        status='starting',
        stage='received',
        reasoning_preview='',
        prompt=prompt[:160]
    )
    try:
        limits = get_generation_limits()
        prompt_content = get_prompt_content()
        if "ERROR" in prompt_content:
            return jsonify({'error': prompt_content}), 500
        soopub_context = get_soopub_context(prompt)

        # Keep prompt.txt as primary instruction source; add contextual notes as secondary reference.
        system_prompt = prompt_content
        if soopub_context:
            system_prompt = (
                f"{prompt_content}\n\n"
                "Additional Organological Context (secondary reference, do not override core prompt rules):\n"
                f"{soopub_context}"
            )
        refactor_clause = ""
        if refact_meta.get('is_refact'):
            refactor_clause = (
                "\n\nRefactor mode (lineage-aware):"
                "\n- The prompt includes BASE context from a previous generation."
                "\n- Preserve instrument identity and iterate on the existing design; do not restart from scratch."
                "\n- Reuse and modify provided matplotlib/trimesh code when available."
                "\n- Return full updated code blocks (matplotlib first, trimesh second), not only patches/diffs."
            )
        plot_output_template = build_plot_output_template()
        # First call: summary + materials + matplotlib only.
        plot_instruction = (
            prompt
            + "\n\nWrite first a concise 120-180 word conceptual summary describing the organogram's design decisions: shapes/schematics, arrows/flows, colors and their meanings, acoustical rationale, organological relations, and performative interactions."
            + "\n\nThen, add a section titled 'Materials' as an enumerated shopping list with quantities and dimensions for building the instrument (e.g., '1 PVC tube, 30 cm length, 2.5 cm diameter'). If part of the instrument is virtual or hybrid, include 'Virtual Materials' items for assets (e.g., textures, shaders, samples)."
            + "\n\nThen, provide the executable Python matplotlib code in a single fenced code block (```python ... ```)."
            + "\n\nStrict generation contract for this first call:"
            + "\n- You are a Python and matplotlib assistant specialized in Mantle Hood organograms."
            + "\n- The first Python block must produce a valid matplotlib organogram figure (no placeholders, no pseudo-code, no blank figure)."
            + "\n- Always include all required imports in each code block."
            + "\n- For arrows and geometric patches use matplotlib.patches and ax.annotate/FancyArrowPatch; avoid plt.Arrow."
            + "\n- Never include the phrase '# Evaluate the text again to render...'."
            + "\n- Keep conceptual summary and materials outside code blocks."
            + "\n- Never answer with prose only."
            + "\n- If uncertain, simplify the organogram and keep the code valid."
            + "\n- The first python block must literally begin with `import matplotlib.pyplot as plt`."
            + "\n\nMandatory output template:"
            + f"\n{plot_output_template}"
            + "\n\nOptional structured blueprint (use when dimensional pipe/component data is available):"
            + "\n- Define dataclass PipeElement with: name, length_mm, diameter_mm, wall_thickness_mm, material, category, acoustic_target."
            + "\n- Implement plot_geometry_vs_frequency(elements) with a 2x2 figure: length vs fundamental_frequency_hz (log-log), diameter vs fundamental_frequency_hz, histogram of wall_thickness_mm, scatter length vs diameter sized by wall_thickness_mm."
            + "\n- Use seaborn-like aesthetics with standard matplotlib only."
            + "\n- Include a __main__ example with 3-4 dummy elements and ensure plotting code is executed so the backend can render the image."
            + refactor_clause
            + "\n\nDo not include any trimesh or STL code in this first call."
            + "\nDo not include any additional commentary after the matplotlib code block."
        )
        request_start = time.perf_counter()
        deadline_at = (
            request_start + float(limits['deadline_sec'])
            if float(limits['deadline_sec']) > 0
            else None
        )
        try:
            (
                raw_response,
                summary_text,
                materials_text,
                plot_code,
                stl_code,
                image_base64,
                stl_bytes,
                llm_meta
            ) = generate_with_image_required(
                system_prompt,
                prompt,
                plot_instruction,
                max_attempts=limits['max_attempts'],
                llm_timeout_sec=limits['llm_timeout_sec'],
                llm_max_tokens=limits['llm_max_tokens'],
                deadline_at=deadline_at,
                plot_output_template=plot_output_template,
                request_id=request_id
            )
        except Exception as first_error:
            can_retry_without_context = bool(soopub_context) and limits.get('retry_without_context', False)
            has_time_budget = deadline_at is None or (deadline_at - time.perf_counter()) > 25
            if can_retry_without_context and has_time_budget:
                logging.error(
                    f"Primary generation with extended context failed: {first_error}. "
                    "Retrying once with prompt.txt only."
                )
                retry_instruction = (
                    plot_instruction
                    + f"\n\nPrevious generation failed with: {first_error}"
                    + "\nRetry with shorter output, but keep full validity of the matplotlib organogram block."
                )
                (
                    raw_response,
                    summary_text,
                    materials_text,
                    plot_code,
                    stl_code,
                    image_base64,
                    stl_bytes,
                    llm_meta
                ) = generate_with_image_required(
                    prompt_content,
                    prompt,
                    retry_instruction,
                    max_attempts=limits['max_attempts'],
                    llm_timeout_sec=limits['llm_timeout_sec'],
                    llm_max_tokens=limits['llm_max_tokens'],
                    deadline_at=deadline_at,
                    plot_output_template=plot_output_template,
                    request_id=request_id
                )
            else:
                raise
        elapsed_ms = int((time.perf_counter() - request_start) * 1000)

        stl_profile = None
        if not stl_bytes:
            try:
                update_generation_progress(request_id, stage='fallback_geometry', status='running')
                stl_bytes, fallback_code, stl_profile = build_fallback_stl_bytes(
                    prompt=prompt,
                    summary_text=summary_text or raw_response,
                    materials_text=materials_text or "",
                    plot_code=plot_code or ""
                )
                stl_code = fallback_code
                logging.warning(
                    f"STL fallback mesh was used because generated trimesh output was unavailable (profile={stl_profile})."
                )
            except Exception as stl_fallback_error:
                logging.error(f"Fallback STL generation failed: {stl_fallback_error}")

        image_bytes = base64.b64decode(image_base64)
        sketch_bytes = None
        sketch_prompt = None
        sketch_model = None
        try:
            update_generation_progress(request_id, stage='sketch', status='running')
            sketch_bytes, sketch_prompt, sketch_model = generate_sketch_image(
                organogram_bytes=image_bytes,
                prompt=prompt,
                prompt_content=prompt_content,
                summary_text=summary_text or raw_response,
                materials_text=materials_text or "",
                plot_code=plot_code or ""
            )
        except Exception as sketch_error:
            logging.error(f"Sketch generation failed: {sketch_error}")

        meta = None
        try:
            meta = _save_gallery_item(
                prompt,
                summary_text or raw_response,
                plot_code or stl_code or '',
                image_bytes,
                stl_bytes=stl_bytes,
                sketch_bytes=sketch_bytes,
                llm_model=llm_meta.get('model'),
                elapsed_ms=elapsed_ms,
                plot_code=plot_code,
                stl_code=stl_code,
                materials_text=materials_text,
                summary_text=summary_text,
                sketch_prompt=sketch_prompt,
                sketch_model=sketch_model,
                refact_meta=refact_meta
            )
            if summary_text:
                meta['summary'] = summary_text
            if materials_text:
                meta['materials_text'] = materials_text
        except Exception as e:
            logging.error(f"Error saving gallery item: {e}")

        image_url = (meta or {}).get('image_url') if isinstance(meta, dict) else None
        sketch_url = (meta or {}).get('sketch_url') if isinstance(meta, dict) else None
        inline_image = image_base64
        inline_sketch = None
        max_inline_image_bytes = int(limits.get('max_inline_image_bytes') or 0)
        if max_inline_image_bytes <= 0:
            inline_image = None
        elif image_url and len(image_bytes) > max_inline_image_bytes:
            inline_image = None
        if sketch_bytes:
            sketch_b64 = base64.b64encode(sketch_bytes).decode('utf-8')
            if max_inline_image_bytes > 0 and len(sketch_bytes) <= max_inline_image_bytes:
                inline_sketch = sketch_b64

        update_generation_progress(
            request_id,
            status='completed',
            stage='done',
            elapsed_ms=elapsed_ms,
            reasoning_preview=''
        )

        return jsonify({
            "type": "plot",
            "content": (plot_code or '').strip(),
            "plot_code": (plot_code or '').strip(),
            "stl_code": (stl_code or '').strip(),
            "image": inline_image,
            "image_url": image_url,
            "sketch": inline_sketch,
            "sketch_url": sketch_url,
            "sketch_prompt": sketch_prompt,
            "sketch_model": sketch_model,
            "gallery": meta,
            "summary": summary_text,
            "materials": materials_text,
            "llm_endpoint": llm_meta.get('endpoint'),
            "llm_model": llm_meta.get('model'),
            "elapsed_ms": elapsed_ms,
            "stl_profile": stl_profile
        })
    except Exception as e:
        logging.error(f"Error in /api/generate: {e}")
        update_generation_progress(
            request_id,
            status='error',
            stage='failed',
            error=str(e)
        )
        return jsonify({
            'error': f'Failed to generate organogram image: {str(e)}. '
                     f'Ensure Ollama model outputs valid matplotlib code.'
        }), 500

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

@app.route('/api/generate/sketch', methods=['POST'])
@log_activity('generate_sketch')
def generate_sketch():
    """
    Generate or regenerate a diffusion sketch based on provided organogram and text.
    Used for instant regeneration from the main results panel.
    """
    data = request.json or {}
    prompt = data.get('prompt', '').strip()
    summary_text = data.get('summary', '').strip()
    materials_text = data.get('materials', '').strip()
    plot_code = data.get('plot_code', '').strip()
    image_b64 = data.get('image', '')

    if not image_b64:
        return jsonify({'error': 'Original organogram image base64 is required'}), 400

    try:
        organogram_bytes = base64.b64decode(image_b64)
        
        sketch_prompt_content = get_specialized_prompt('inferred-image')
        sketch_bytes, sketch_prompt, sketch_model = generate_sketch_image(
            organogram_bytes=organogram_bytes,
            prompt=prompt,
            prompt_content=sketch_prompt_content,
            summary_text=summary_text,
            materials_text=materials_text,
            plot_code=plot_code
        )

        if not sketch_bytes:
            return jsonify({'error': 'Sketch generation failed'}), 500

        sketch_b64 = base64.b64encode(sketch_bytes).decode('utf-8')
        return jsonify({
            'ok': True,
            'sketch': sketch_b64,
            'sketch_prompt': sketch_prompt,
            'sketch_model': sketch_model
        })
    except Exception as e:
        logging.error(f"Error in /api/generate/sketch: {e}")
        return jsonify({'error': str(e)}), 500
@app.route('/api/gallery/item/<basename>/remake_sketch', methods=['POST'])
@log_activity('remake_sketch')
def remake_sketch(basename: str):
    """
    Regenerate the diffusion sketch for an existing gallery item.
    Uses prompt and summary from the JSON metadata.
    """
    if not basename:
        return jsonify({'error': 'Missing basename'}), 400

    request_id = request.json.get('request_id') if (request.is_json and request.json) else None

    meta_path = os.path.join(GALLERY_DIR, f"{basename}.json")
    if not os.path.isfile(meta_path):
        return jsonify({'error': 'Gallery item not found'}), 404

    try:
        update_generation_progress(request_id, stage='loading_metadata', status='running')
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)

        prompt = meta.get('prompt', '') or ''
        summary_text = meta.get('summary', '') or meta.get('answer', '') or ''
        materials_text = meta.get('materials_text', '') or ''
        plot_code = meta.get('plot_code', '') or ''
        image_url = meta.get('image_url', '') or ''

        if not image_url:
            return jsonify({'error': 'Original organogram image not found in metadata'}), 400

        # Load original image bytes
        image_rel_path = image_url
        if image_rel_path.startswith('/offload/'):
             image_rel_path = image_rel_path[len('/offload/'):]
        elif image_rel_path.startswith('/api/gallery/image/'):
             image_rel_path = os.path.join('gallery', image_rel_path[len('/api/gallery/image/'):])
        
        full_image_path = os.path.join(OFFLOAD_DIR, image_rel_path)
        if not os.path.isfile(full_image_path):
            return jsonify({'error': f'Original organogram image file not found: {full_image_path}'}), 404

        with open(full_image_path, 'rb') as f:
            organogram_bytes = f.read()

        # Generate new sketch
        update_generation_progress(request_id, stage='regenerating_sketch', status='running')
        sketch_prompt_content = get_specialized_prompt('inferred-image')
        sketch_bytes, sketch_prompt, sketch_model = generate_sketch_image(
            organogram_bytes=organogram_bytes,
            prompt=prompt,
            prompt_content=sketch_prompt_content,
            summary_text=summary_text,
            materials_text=materials_text,
            plot_code=plot_code
        )

        if not sketch_bytes:
             return jsonify({'error': 'Sketch generation failed to produce output'}), 500

        # Save new sketch file
        sketch_filename = f"{basename}.sketch.png"
        sketch_path = os.path.join(GALLERY_DIR, sketch_filename)
        with open(sketch_path, 'wb') as f:
            f.write(sketch_bytes)

        # Update metadata
        meta['sketch_url'] = f"/api/gallery/image/{sketch_filename}"
        meta['sketch_prompt'] = sketch_prompt
        meta['sketch_model'] = sketch_model
        meta['updated_at'] = datetime.datetime.now().isoformat()

        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        update_generation_progress(request_id, stage='completed', status='completed')
        return jsonify({
            'ok': True,
            'sketch_url': meta['sketch_url'],
            'sketch_prompt': sketch_prompt,
            'sketch_model': sketch_model,
            'item': meta
        })

    except Exception as e:
        logging.error(f"Error remaking sketch for {basename}: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/gallery/item/<basename>/generate_sound', methods=['POST'])
@log_activity('generate_sound')
def generate_sound(basename: str):
    """
    Generate audio samples for an existing gallery item using Stable Audio Open.
    """
    if not basename:
        return jsonify({'error': 'Missing basename'}), 400

    request_id = request.json.get('request_id') if request.is_json else None

    meta_path = os.path.join(GALLERY_DIR, f"{basename}.json")
    if not os.path.isfile(meta_path):
        return jsonify({'error': 'Gallery item not found'}), 404

    try:
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)

        prompt = meta.get('prompt', '') or ''
        summary_text = meta.get('summary', '') or meta.get('answer', '') or ''
        materials_text = meta.get('materials_text', '') or ''

        # Generate sound samples
        sound_results, sound_model = generate_sound_samples(
            prompt=prompt,
            summary_text=summary_text,
            materials_text=materials_text,
            basename=basename,
            request_id=request_id
        )

        if not sound_results:
            return jsonify({'error': 'Sound generation failed to produce output'}), 500

        # Update metadata
        meta['sound_samples'] = sound_results
        meta['sound_model'] = sound_model
        meta['updated_at'] = datetime.datetime.now().isoformat()

        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        return jsonify({
            'ok': True,
            'sound_samples': sound_results,
            'sound_model': sound_model,
            'item': meta
        })

    except Exception as e:
        logging.error(f"Error generating sound for {basename}: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/version', methods=['GET'])
@log_activity('version')
def version():
    try:
        v = get_version()
        response = jsonify({'version': v})
        response.headers['Content-Type'] = 'application/json'
        return response
    except Exception as e:
        logging.error(f"Error in /api/version: {e}")
        error_response = jsonify({'version': '0.0.0', 'error': str(e)})
        error_response.headers['Content-Type'] = 'application/json'
        return error_response, 500


@app.route('/api/health', methods=['GET'])
@log_activity('health')
def health():
    """Simple health check with version and device info"""
    try:
        base_url, model_name, _ = get_ollama_config()
        return jsonify({
            'status': 'ok',
            'version': get_version(),
            'cuda': torch.cuda.is_available(),
            'device': str(device),
            'llm_provider': 'ollama',
            'ollama': {
                'configured': bool(model_name),
                'base_url': base_url,
                'model': model_name,
                'override_active': bool(OLLAMA_MODEL_OVERRIDE)
            }
        })
    except Exception as e:
        logging.error(f"Error in /api/health: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/somap/graph', methods=['GET'])
@log_activity('somap-graph')
def somap_graph():
    """
    Build a knowledge graph from markdown notes in backend/soopub:
    folders, notes, tags, and Obsidian wikilinks.
    """
    try:
        graph = _build_soopub_graph()
        return jsonify(graph)
    except Exception as e:
        logging.error(f"Error in /api/somap/graph: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/ollama/verify', methods=['GET'])
@app.route('/api/deepseek/verify', methods=['GET'])
@log_activity('ollama-verify')
def ollama_verify():
    """Verify Ollama connectivity by listing available models."""
    try:
        res = fetch_ollama_models()
        if res.get('ok'):
            models = res.get('models', [])
            configured = res.get('configured_model')
            return jsonify({
                'ok': True,
                'endpoint': res.get('endpoint'),
                'configured_model': configured,
                'model_available': configured in models,
                'model_count': len(models),
                'models': models[:50]
            })
        return jsonify({
            'ok': False,
            'status': res.get('status', 500),
            'error': res.get('error', 'Unknown Ollama error')
        }), 502
    except Exception as e:
        logging.error(f"Error in /api/ollama/verify: {e}")
        return jsonify({'ok': False, 'error': str(e)}), 500


@app.route('/api/ollama/models', methods=['GET'])
@log_activity('ollama-models')
def ollama_models():
    """List installed Ollama models and current active model."""
    try:
        res = fetch_ollama_models()
        if not res.get('ok'):
            return jsonify({
                'ok': False,
                'error': res.get('error', 'Could not list Ollama models'),
                'status': res.get('status', 500)
            }), 502

        models = res.get('models', [])
        current = res.get('configured_model')
        return jsonify({
            'ok': True,
            'models': models,
            'current_model': current,
            'model_available': current in models,
            'override_active': bool(OLLAMA_MODEL_OVERRIDE)
        })
    except Exception as e:
        logging.error(f"Error in /api/ollama/models: {e}")
        return jsonify({'ok': False, 'error': str(e)}), 500


@app.route('/api/ollama/model', methods=['POST'])
@log_activity('ollama-model-set')
def ollama_set_model():
    """Set current Ollama model at runtime for this backend process."""
    global OLLAMA_MODEL_OVERRIDE
    try:
        body = request.get_json(force=True) or {}
        requested = (body.get('model') or '').strip()
        if not requested:
            return jsonify({'ok': False, 'error': 'Missing model field'}), 400

        res = fetch_ollama_models()
        if not res.get('ok'):
            return jsonify({
                'ok': False,
                'error': res.get('error', 'Could not list Ollama models')
            }), 502

        models = res.get('models', [])
        if requested not in models:
            return jsonify({
                'ok': False,
                'error': f"Model '{requested}' is not installed",
                'models': models[:50]
            }), 400

        OLLAMA_MODEL_OVERRIDE = requested
        return jsonify({
            'ok': True,
            'model': requested,
            'override_active': True
        })
    except Exception as e:
        logging.error(f"Error in /api/ollama/model: {e}")
        return jsonify({'ok': False, 'error': str(e)}), 500


@app.route('/api/gallery/list', methods=['GET'])
def gallery_list():
    try:
        items = []
        if not os.path.isdir(GALLERY_DIR):
            return jsonify({'items': []})
        for name in os.listdir(GALLERY_DIR):
            if name.endswith('.json'):
                path = os.path.join(GALLERY_DIR, name)
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        meta = json.load(f)
                        title = (meta.get('title') or '').strip() or _title_from_basename(meta.get('basename', ''))
                        meta['title'] = title or 'untitled'
                        meta['title_slug'] = (meta.get('title_slug') or '').strip() or _safe_slug(meta['title'], fallback='untitled')
                        meta['group_id'] = _meta_group_id(meta) or meta.get('basename', '')
                        version_index = _version_index_from_value(
                            meta.get('version_index') or meta.get('version'),
                            default=1
                        )
                        meta['version_index'] = version_index
                        meta['version'] = (meta.get('version') or '').strip() or _version_label_from_index(version_index)
                        items.append(meta)
                except Exception:
                    continue
        # sort by timestamp desc
        items.sort(key=lambda m: m.get('timestamp', ''), reverse=True)
        return jsonify({'items': items})
    except Exception as e:
        logging.error(f"Error listing gallery: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/gallery/image/<path:filename>', methods=['GET'])
def gallery_image(filename):
    try:
        return send_from_directory(GALLERY_DIR, filename)
    except Exception as e:
        logging.error(f"Error serving gallery image {filename}: {e}")
        return jsonify({'error': 'Image not found'}), 404


@app.route('/api/gallery/file/<path:filename>', methods=['GET'])
def gallery_file(filename):
    """Serve arbitrary gallery files (e.g., STL)."""
    try:
        return send_from_directory(GALLERY_DIR, filename, as_attachment=True)
    except Exception as e:
        logging.error(f"Error serving gallery file {filename}: {e}")
        return jsonify({'error': 'File not found'}), 404


@app.route('/api/gallery/item/<string:basename>', methods=['DELETE'])
def gallery_delete_item(basename):
    """Delete a gallery item by its basename."""
    try:
        # Basic validation on basename
        if not re.match(r'^[a-zA-Z0-9_\-]+$', basename):
            return jsonify({'error': 'Invalid basename format'}), 400

        files_to_delete = []
        for ext in GALLERY_FILE_SUFFIXES:
            filename = f"{basename}{ext}"
            path = os.path.join(GALLERY_DIR, filename)
            if os.path.exists(path):
                files_to_delete.append(path)

        if not files_to_delete:
            return jsonify({'error': 'Item not found'}), 404

        for path in files_to_delete:
            try:
                os.remove(path)
            except OSError as e:
                logging.warning(f"Could not delete file {path}: {e}")
                # Decide if you want to fail the whole operation or just log
                # For robustness, we log and continue

        return jsonify({'ok': True, 'message': f'Item {basename} deleted'})
    except Exception as e:
        logging.error(f"Error deleting gallery item {basename}: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/gallery/item/<string:basename>/rename', methods=['POST'])
def gallery_rename_item(basename):
    """Rename a gallery item."""
    try:
        data = request.json
        new_name = data.get('newName', '').strip()

        if not new_name or not re.match(r'^[a-zA-Z0-9_\-]+$', new_name):
            return jsonify({'error': 'Invalid new name format'}), 400
        
        if not re.match(r'^[a-zA-Z0-9_\-]+$', basename):
            return jsonify({'error': 'Invalid basename format'}), 400

        timestamp = basename.split('_')[0]
        new_basename = f"{timestamp}_{new_name}"

        # Check if new name already exists
        if os.path.exists(os.path.join(GALLERY_DIR, f"{new_basename}.json")):
            return jsonify({'error': 'New name already exists'}), 409

        # Rename files
        renamed_files = []
        for ext in GALLERY_FILE_SUFFIXES:
            old_path = os.path.join(GALLERY_DIR, f"{basename}{ext}")
            if os.path.exists(old_path):
                new_path = os.path.join(GALLERY_DIR, f"{new_basename}{ext}")
                os.rename(old_path, new_path)
                renamed_files.append(new_path)
        
        if not renamed_files:
            return jsonify({'error': 'Item not found'}), 404

        # Update JSON metadata
        json_path = os.path.join(GALLERY_DIR, f"{new_basename}.json")
        if os.path.exists(json_path):
            with open(json_path, 'r+') as f:
                meta = json.load(f)
                meta['basename'] = new_basename
                if meta.get('image_url'):
                    meta['image_url'] = f"/api/gallery/image/{new_basename}.png"
                if meta.get('stl_url'):
                    meta['stl_url'] = f"/api/gallery/file/{new_basename}.stl"
                if meta.get('sketch_url'):
                    meta['sketch_url'] = f"/api/gallery/image/{new_basename}.sketch.png"
                
                f.seek(0)
                json.dump(meta, f)
                f.truncate()

        return jsonify({'ok': True, 'message': f'Item {basename} renamed to {new_basename}', 'newBasename': new_basename})

    except Exception as e:
        logging.error(f"Error renaming gallery item {basename}: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/gallery/group/<string:group_id>/rename', methods=['POST'])
def gallery_rename_group(group_id):
    """Rename all versions in the same gallery group."""
    try:
        if not re.match(r'^[a-zA-Z0-9_\-]+$', group_id):
            return jsonify({'error': 'Invalid group format'}), 400

        data = request.get_json(force=True) or {}
        new_name = (data.get('newName') or '').strip()
        if not new_name:
            return jsonify({'error': 'New name is required'}), 400

        new_slug = _safe_slug(new_name, fallback='untitled')
        new_title = new_slug.replace('_', '-')

        selected = []
        for name in os.listdir(GALLERY_DIR):
            if not name.endswith('.json'):
                continue
            path = os.path.join(GALLERY_DIR, name)
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
            except Exception:
                continue
            if _meta_group_id(meta) != group_id:
                continue
            selected.append(meta)

        if not selected:
            return jsonify({'error': 'Group not found'}), 404

        selected.sort(key=lambda meta: meta.get('timestamp', ''))
        used_basenames = {meta.get('basename', '') for meta in selected}
        existing_json_files = {
            os.path.splitext(name)[0]
            for name in os.listdir(GALLERY_DIR)
            if name.endswith('.json')
        }

        updated = []
        for meta in selected:
            old_base = meta.get('basename', '')
            if not old_base:
                continue
            timestamp = str(meta.get('timestamp') or old_base.split('_')[0])
            version_token = _safe_slug(str(meta.get('version') or 'v1'), fallback='v1')
            candidate = f"{timestamp}_{new_slug}_{version_token}"
            suffix = 1
            while candidate in existing_json_files and candidate not in used_basenames:
                candidate = f"{timestamp}_{new_slug}_{version_token}_{suffix}"
                suffix += 1

            new_base = candidate
            existing_json_files.add(new_base)
            used_basenames.add(new_base)

            # Rename physical files
            for ext in GALLERY_FILE_SUFFIXES:
                old_path = os.path.join(GALLERY_DIR, f"{old_base}{ext}")
                if not os.path.exists(old_path):
                    continue
                new_path = os.path.join(GALLERY_DIR, f"{new_base}{ext}")
                os.rename(old_path, new_path)

            json_path = os.path.join(GALLERY_DIR, f"{new_base}.json")
            if os.path.exists(json_path):
                with open(json_path, 'r+', encoding='utf-8') as f:
                    current = json.load(f)
                    current['basename'] = new_base
                    current['title'] = new_title
                    current['title_slug'] = new_slug
                    if current.get('image_url'):
                        current['image_url'] = f"/api/gallery/image/{new_base}.png"
                    if current.get('stl_url'):
                        current['stl_url'] = f"/api/gallery/file/{new_base}.stl"
                    if current.get('sketch_url'):
                        current['sketch_url'] = f"/api/gallery/image/{new_base}.sketch.png"
                    f.seek(0)
                    json.dump(current, f)
                    f.truncate()

            updated.append({'old': old_base, 'new': new_base})

        return jsonify({
            'ok': True,
            'group_id': group_id,
            'title': new_title,
            'title_slug': new_slug,
            'updated': updated
        })
    except Exception as e:
        logging.error(f"Error renaming gallery group {group_id}: {e}")
        return jsonify({'error': str(e)}), 500





@app.route('/api/gallery/item/<string:basename>/featured', methods=['POST'])
def gallery_featured_item(basename):
    """Toggle featured status of a gallery item."""
    try:
        data = request.get_json(force=True) or {}
        featured = bool(data.get('featured', False))

        json_path = os.path.join(GALLERY_DIR, f"{basename}.json")
        if not os.path.exists(json_path):
            return jsonify({'error': 'Item not found'}), 404

        with open(json_path, 'r+', encoding='utf-8') as f:
            meta = json.load(f)
            meta['featured'] = featured
            f.seek(0)
            json.dump(meta, f, ensure_ascii=False, indent=2)
            f.truncate()

        # Also update the .txt file if needed, but JSON is the primary source for the gallery list.
        txt_path = os.path.join(GALLERY_DIR, f"{basename}.txt")
        if os.path.exists(txt_path):
            with open(txt_path, 'r+', encoding='utf-8') as f:
                content = f.read()
                # Simple way to add/update featured in the metadata section of the text file
                if "# Generation Metadata" in content:
                    if "- featured:" in content:
                        content = re.sub(r'- featured:.*', f'- featured: {featured}', content)
                    else:
                        content = content.replace("# Generation Metadata\n\n", f"# Generation Metadata\n\n- featured: {featured}\n")
                f.seek(0)
                f.write(content)
                f.truncate()

        return jsonify({'ok': True, 'basename': basename, 'featured': featured})
    except Exception as e:
        logging.error(f"Error updating featured status for {basename}: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/dev/save', methods=['POST'])
def dev_save():
    # Restricted in production unless explicitly enabled
    if not app.debug and os.getenv('ENABLE_DEV_SAVE', '') != '1':
        return jsonify({'error': 'dev_save disabled'}), 403
    try:
        data = request.get_json(force=True) or {}
        prompt = (data.get('prompt') or 'dev test').strip()
        summary = (data.get('summary') or '').strip()
        matplotlib_code = data.get('matplotlib_code') or ''
        trimesh_code = data.get('trimesh_code') or ''

        image_bytes = b''
        stl_bytes = None
        plot_code = None
        stl_code = None

        if matplotlib_code.strip():
            try:
                image_b64 = execute_matplotlib_code(matplotlib_code)
                image_bytes = base64.b64decode(image_b64)
                plot_code = matplotlib_code
            except Exception as e:
                logging.error(f"Dev matplotlib exec failed: {e}")
        if trimesh_code.strip():
            try:
                stl_bytes = execute_trimesh_code(trimesh_code)
                stl_code = trimesh_code
            except Exception as e:
                logging.error(f"Dev trimesh exec failed: {e}")

        if not image_bytes and not stl_bytes:
            return jsonify({'error': 'No outputs produced'}), 400

        code_to_save = plot_code or stl_code or ''
        meta = _save_gallery_item(prompt, summary, code_to_save, image_bytes, stl_bytes=stl_bytes)
        return jsonify({'ok': True, 'gallery': meta})
    except Exception as e:
        logging.error(f"Error in /api/dev/save: {e}")
        return jsonify({'error': str(e)}), 500


@app.errorhandler(404)
def page_not_found(e):
    logging.warning(f"404 Not Found: {request.method} {request.url}")
    return jsonify(error="Not Found", url=request.url), 404

if __name__ == '__main__':
    port = int(os.getenv("PORT", 10000))
    debug_mode = os.getenv("FLASK_DEBUG", "1").strip().lower() in ("1", "true", "yes", "on")
    use_reloader = os.getenv("FLASK_RELOAD", "1").strip().lower() in ("1", "true", "yes", "on")
    logging.info(
        f"Starting Flask app on port {port} "
        f"(debug={debug_mode}, reloader={use_reloader})"
    )
    app.run(host='0.0.0.0', port=port, debug=debug_mode, use_reloader=use_reloader)
