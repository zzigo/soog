from functools import wraps
from cache_manager import cache
from cache_config import CacheType, get_gpt_cache_key, get_plot_cache_key, get_model_cache_key
import logging
from typing import Callable, Any

def cache_gpt_response(f: Callable) -> Callable:
    """Decorator to cache GPT API responses"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            # Extract prompt and system prompt from request
            prompt = kwargs.get('prompt', '')
            system_prompt = kwargs.get('system_prompt', '')
            
            # Generate cache key
            cache_key = get_gpt_cache_key(prompt, system_prompt)
            
            # Try to get from cache
            cached_response = cache.get(cache_key)
            if cached_response:
                logging.info(f"Cache hit for GPT response: {cache_key}")
                return cached_response
            
            # If not in cache, call original function
            response = f(*args, **kwargs)
            
            # Cache the response
            cache.set(cache_key, response, CacheType.GPT_RESPONSE)
            logging.info(f"Cached GPT response: {cache_key}")
            
            return response
        except Exception as e:
            logging.error(f"Error in cache_gpt_response decorator: {e}")
            return f(*args, **kwargs)  # Fallback to original function
    return decorated_function

def cache_plot(f: Callable) -> Callable:
    """Decorator to cache matplotlib plots"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            # Extract plot code
            code = kwargs.get('code', '')
            if not code:
                return f(*args, **kwargs)
            
            # Generate cache key
            cache_key = get_plot_cache_key(code)
            
            # Try to get from cache
            cached_plot = cache.get(cache_key)
            if cached_plot:
                logging.info(f"Cache hit for plot: {cache_key}")
                return cached_plot
            
            # If not in cache, generate plot
            plot_data = f(*args, **kwargs)
            
            # Cache the plot
            cache.set(cache_key, plot_data, CacheType.PLOT_IMAGE)
            logging.info(f"Cached plot: {cache_key}")
            
            return plot_data
        except Exception as e:
            logging.error(f"Error in cache_plot decorator: {e}")
            return f(*args, **kwargs)  # Fallback to original function
    return decorated_function

def cache_model(model_name: str) -> Any:
    """Function to handle model caching"""
    try:
        cache_key = get_model_cache_key(model_name)
        
        # Try to get from cache
        cached_model = cache.get(cache_key, model_cache=True)
        if cached_model:
            logging.info(f"Cache hit for model: {cache_key}")
            return cached_model
        
        return None
    except Exception as e:
        logging.error(f"Error in cache_model: {e}")
        return None

def save_model_to_cache(model_name: str, model: Any) -> bool:
    """Function to save model to cache"""
    try:
        cache_key = get_model_cache_key(model_name)
        success = cache.set(cache_key, model, CacheType.MODEL_CACHE, model_cache=True)
        if success:
            logging.info(f"Cached model: {cache_key}")
        return success
    except Exception as e:
        logging.error(f"Error saving model to cache: {e}")
        return False
