import redis
import pickle
from typing import Any, Optional, Union
from cache_config import REDIS_CONFIG, CACHE_TTLS, CacheType
import logging

class CacheManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(CacheManager, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Initialize Redis connections"""
        try:
            self.default_client = redis.Redis(**REDIS_CONFIG['default'])
            self.model_client = redis.Redis(**REDIS_CONFIG['model_cache'])
            logging.info("Redis connections established")
        except redis.ConnectionError as e:
            logging.error(f"Failed to connect to Redis: {e}")
            raise

    def get(self, key: str, model_cache: bool = False) -> Optional[Any]:
        """Get value from cache"""
        try:
            client = self.model_client if model_cache else self.default_client
            value = client.get(key)
            if value is None:
                return None
            return pickle.loads(value) if model_cache else value
        except Exception as e:
            logging.error(f"Error retrieving from cache: {e}")
            return None

    def set(self, key: str, value: Any, cache_type: CacheType, model_cache: bool = False) -> bool:
        """Set value in cache with appropriate TTL"""
        try:
            client = self.model_client if model_cache else self.default_client
            ttl = CACHE_TTLS[cache_type].total_seconds()
            
            # Pickle objects for model cache, use raw value for default cache
            cache_value = pickle.dumps(value) if model_cache else value
            return client.setex(key, int(ttl), cache_value)
        except Exception as e:
            logging.error(f"Error setting cache: {e}")
            return False

    def delete(self, key: str, model_cache: bool = False) -> bool:
        """Delete key from cache"""
        try:
            client = self.model_client if model_cache else self.default_client
            return client.delete(key) > 0
        except Exception as e:
            logging.error(f"Error deleting from cache: {e}")
            return False

    def clear_cache_type(self, pattern: str, model_cache: bool = False) -> bool:
        """Clear all keys matching a pattern"""
        try:
            client = self.model_client if model_cache else self.default_client
            keys = client.keys(pattern)
            if keys:
                return client.delete(*keys) > 0
            return True
        except Exception as e:
            logging.error(f"Error clearing cache: {e}")
            return False

    def get_cache_stats(self) -> dict:
        """Get cache statistics"""
        try:
            default_info = self.default_client.info()
            model_info = self.model_client.info()
            return {
                'default_db': {
                    'used_memory': default_info['used_memory_human'],
                    'connected_clients': default_info['connected_clients'],
                    'total_keys': self.default_client.dbsize()
                },
                'model_db': {
                    'used_memory': model_info['used_memory_human'],
                    'connected_clients': model_info['connected_clients'],
                    'total_keys': self.model_client.dbsize()
                }
            }
        except Exception as e:
            logging.error(f"Error getting cache stats: {e}")
            return {}

    def close(self):
        """Close Redis connections"""
        try:
            self.default_client.close()
            self.model_client.close()
            logging.info("Redis connections closed")
        except Exception as e:
            logging.error(f"Error closing Redis connections: {e}")

# Global cache manager instance
cache = CacheManager()
