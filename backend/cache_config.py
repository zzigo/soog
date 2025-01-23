from enum import Enum
from datetime import timedelta

class CacheType(Enum):
    GPT_RESPONSE = "gpt_response"
    PLOT_IMAGE = "plot_image"
    MODEL_CACHE = "model_cache"
    STATIC_CONTENT = "static_content"

# Redis Configuration
REDIS_CONFIG = {
    'default': {
        'host': 'localhost',
        'port': 6379,
        'db': 0,
        'decode_responses': True
    },
    'model_cache': {  # Separate DB for large model objects
        'host': 'localhost',
        'port': 6379,
        'db': 1,
        'decode_responses': False
    }
}

# Cache TTLs
CACHE_TTLS = {
    CacheType.GPT_RESPONSE: timedelta(hours=1),
    CacheType.PLOT_IMAGE: timedelta(minutes=30),
    CacheType.MODEL_CACHE: timedelta(hours=24),
    CacheType.STATIC_CONTENT: timedelta(hours=12)
}

# Redis Memory Configuration
CACHE_CONFIG = {
    'maxmemory': '2gb',
    'maxmemory-policy': 'allkeys-lru',  # Least Recently Used eviction
    'maxmemory-samples': 5
}

# Cache Keys
def get_gpt_cache_key(prompt: str, system_prompt: str) -> str:
    """Generate cache key for GPT responses"""
    import hashlib
    combined = f"{prompt}:{system_prompt}"
    return f"gpt:{hashlib.md5(combined.encode()).hexdigest()}"

def get_plot_cache_key(code: str) -> str:
    """Generate cache key for matplotlib plots"""
    import hashlib
    return f"plot:{hashlib.md5(code.encode()).hexdigest()}"

def get_model_cache_key(model_name: str) -> str:
    """Generate cache key for ML models"""
    return f"model:{model_name}"

def get_static_cache_key(content_type: str) -> str:
    """Generate cache key for static content"""
    return f"static:{content_type}"
