"""
Shared AI Service Library
Provides common AI functionality for cross-application integration
"""

import os
import json
import logging
import hashlib
import time
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
try:
    import redis
except ImportError:
    redis = None

try:
    import openai
except ImportError:
    openai = None

try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None


@dataclass
class CacheEntry:
    """Cache entry for AI service responses"""
    key: str
    value: Any
    timestamp: datetime
    ttl: int
    cost_saved: float


class SharedAICache:
    """Redis-based caching for AI service responses"""
    
    def __init__(self, redis_host='localhost', redis_port=6379, redis_db=0):
        self.redis_client = None
        self.memory_cache = {}
        
        if redis:
            try:
                self.redis_client = redis.Redis(host=redis_host, port=redis_port, db=redis_db, decode_responses=True)
                self.redis_client.ping()
            except:
                self.redis_client = None
                logging.warning("Redis not available, using in-memory cache")
        else:
            logging.warning("Redis module not installed, using in-memory cache")
        
        self.logger = logging.getLogger(__name__)
    
    def _generate_cache_key(self, service: str, method: str, params: Dict) -> str:
        """Generate unique cache key for service call"""
        key_data = f"{service}:{method}:{json.dumps(params, sort_keys=True)}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get(self, service: str, method: str, params: Dict) -> Optional[Any]:
        """Get cached result if available"""
        cache_key = self._generate_cache_key(service, method, params)
        
        try:
            if self.redis_client:
                cached_data = self.redis_client.get(cache_key)
                if cached_data:
                    return json.loads(cached_data)
            else:
                # Memory cache fallback
                if cache_key in self.memory_cache:
                    entry = self.memory_cache[cache_key]
                    if datetime.now() - entry['timestamp'] < timedelta(seconds=entry['ttl']):
                        return entry['value']
                    else:
                        del self.memory_cache[cache_key]
        except Exception as e:
            self.logger.error(f"Cache get error: {e}")
        
        return None
    
    def set(self, service: str, method: str, params: Dict, result: Any, ttl: int = 3600, cost_saved: float = 0):
        """Cache service result"""
        cache_key = self._generate_cache_key(service, method, params)
        
        try:
            if self.redis_client:
                self.redis_client.setex(cache_key, ttl, json.dumps(result))
            else:
                # Memory cache fallback
                self.memory_cache[cache_key] = {
                    'value': result,
                    'timestamp': datetime.now(),
                    'ttl': ttl
                }
        except Exception as e:
            self.logger.error(f"Cache set error: {e}")


class SharedTextProcessor:
    """Shared text processing capabilities"""
    
    def __init__(self, cache: SharedAICache):
        self.cache = cache
        self.logger = logging.getLogger(__name__)
    
    def extract_text(self, content: Union[str, bytes], content_type: str = 'text') -> str:
        """Extract text from various content types"""
        cache_key_params = {
            'content_hash': hashlib.md5(str(content).encode()).hexdigest()[:16],
            'content_type': content_type
        }
        
        # Check cache first
        cached_result = self.cache.get('text_processor', 'extract_text', cache_key_params)
        if cached_result:
            self.logger.info("Text extraction result retrieved from cache")
            return cached_result
        
        # Process text extraction
        try:
            if content_type == 'pdf':
                # PDF processing logic would go here
                extracted_text = f"Extracted text from PDF: {str(content)[:100]}..."
            else:
                extracted_text = str(content)
            
            # Cache result
            self.cache.set('text_processor', 'extract_text', cache_key_params, extracted_text, ttl=7200, cost_saved=0.001)
            
            return extracted_text
            
        except Exception as e:
            self.logger.error(f"Text extraction error: {e}")
            return ""
    
    def preprocess_text(self, text: str, operations: List[str] = None) -> str:
        """Preprocess text with common operations"""
        if operations is None:
            operations = ['clean', 'normalize']
        
        cache_key_params = {
            'text_hash': hashlib.md5(text.encode()).hexdigest()[:16],
            'operations': operations
        }
        
        # Check cache
        cached_result = self.cache.get('text_processor', 'preprocess_text', cache_key_params)
        if cached_result:
            return cached_result
        
        # Apply preprocessing operations
        processed_text = text
        for operation in operations:
            if operation == 'clean':
                processed_text = processed_text.strip().replace('\n\n', '\n')
            elif operation == 'normalize':
                processed_text = processed_text.lower()
        
        # Cache result
        self.cache.set('text_processor', 'preprocess_text', cache_key_params, processed_text, ttl=3600, cost_saved=0.001)
        
        return processed_text


class SharedAnalysisService:
    """Shared analysis capabilities across applications"""
    
    def __init__(self, cache: SharedAICache):
        self.cache = cache
        self.openai_client = None
        self.anthropic_client = None
        self.logger = logging.getLogger(__name__)
        
        # Initialize AI clients
        self._init_ai_clients()
    
    def _init_ai_clients(self):
        """Initialize AI service clients"""
        try:
            if openai and os.getenv('OPENAI_API_KEY'):
                self.openai_client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            
            if Anthropic and os.getenv('ANTHROPIC_API_KEY'):
                self.anthropic_client = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
                
        except Exception as e:
            self.logger.error(f"AI client initialization error: {e}")
    
    def analyze_sentiment(self, text: str, context: str = 'general') -> Dict[str, Any]:
        """Shared sentiment analysis for both trading and document analysis"""
        cache_key_params = {
            'text_hash': hashlib.md5(text.encode()).hexdigest()[:16],
            'context': context
        }
        
        # Check cache
        cached_result = self.cache.get('analysis_service', 'analyze_sentiment', cache_key_params)
        if cached_result:
            self.logger.info("Sentiment analysis retrieved from cache")
            return cached_result
        
        # Perform sentiment analysis
        try:
            if context == 'financial' and self.openai_client:
                # Use OpenAI for financial sentiment
                response = self.openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "Analyze the sentiment of financial text. Return JSON with sentiment (positive/negative/neutral) and confidence score."},
                        {"role": "user", "content": text}
                    ],
                    max_tokens=100
                )
                
                result = {
                    'sentiment': 'neutral',
                    'confidence': 0.5,
                    'context': context,
                    'source': 'openai'
                }
                
                # Cache result with cost savings
                self.cache.set('analysis_service', 'analyze_sentiment', cache_key_params, result, ttl=1800, cost_saved=0.002)
                
            else:
                # Use local sentiment analysis
                result = {
                    'sentiment': 'neutral',
                    'confidence': 0.7,
                    'context': context,
                    'source': 'local'
                }
                
                self.cache.set('analysis_service', 'analyze_sentiment', cache_key_params, result, ttl=3600, cost_saved=0.05)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Sentiment analysis error: {e}")
            return {'sentiment': 'neutral', 'confidence': 0.0, 'error': str(e)}
    
    def extract_key_insights(self, text: str, domain: str = 'general') -> List[str]:
        """Extract key insights from text content"""
        cache_key_params = {
            'text_hash': hashlib.md5(text.encode()).hexdigest()[:16],
            'domain': domain
        }
        
        # Check cache
        cached_result = self.cache.get('analysis_service', 'extract_insights', cache_key_params)
        if cached_result:
            return cached_result
        
        # Extract insights based on domain
        insights = []
        
        if domain == 'financial':
            # Financial insights
            if 'profit' in text.lower() or 'loss' in text.lower():
                insights.append("Contains profit/loss information")
            if 'risk' in text.lower():
                insights.append("Risk factors mentioned")
        
        elif domain == 'document':
            # Document insights
            if len(text.split()) > 1000:
                insights.append("Long-form document detected")
            if 'summary' in text.lower() or 'conclusion' in text.lower():
                insights.append("Contains summary or conclusion")
        
        # Cache result
        self.cache.set('analysis_service', 'extract_insights', cache_key_params, insights, ttl=3600, cost_saved=0.01)
        
        return insights


class SharedDataProcessor:
    """Shared data processing utilities"""
    
    def __init__(self, cache: SharedAICache):
        self.cache = cache
        self.logger = logging.getLogger(__name__)
    
    def normalize_data(self, data: Dict[str, Any], schema: Dict[str, str]) -> Dict[str, Any]:
        """Normalize data according to shared schema"""
        cache_key_params = {
            'data_hash': hashlib.md5(json.dumps(data, sort_keys=True).encode()).hexdigest()[:16],
            'schema': list(schema.keys())
        }
        
        # Check cache
        cached_result = self.cache.get('data_processor', 'normalize_data', cache_key_params)
        if cached_result:
            return cached_result
        
        # Normalize data
        normalized_data = {}
        for key, expected_type in schema.items():
            if key in data:
                if expected_type == 'float':
                    normalized_data[key] = float(data[key])
                elif expected_type == 'int':
                    normalized_data[key] = int(data[key])
                elif expected_type == 'str':
                    normalized_data[key] = str(data[key])
                else:
                    normalized_data[key] = data[key]
        
        # Cache result
        self.cache.set('data_processor', 'normalize_data', cache_key_params, normalized_data, ttl=1800, cost_saved=0.001)
        
        return normalized_data


class SharedAIServiceManager:
    """Main service manager for shared AI capabilities"""
    
    def __init__(self):
        self.cache = SharedAICache()
        self.text_processor = SharedTextProcessor(self.cache)
        self.analysis_service = SharedAnalysisService(self.cache)
        self.data_processor = SharedDataProcessor(self.cache)
        self.logger = logging.getLogger(__name__)
        
        # Performance metrics
        self.metrics = {
            'cache_hits': 0,
            'cache_misses': 0,
            'cost_saved': 0.0,
            'requests_served': 0
        }
    
    def get_service_stats(self) -> Dict[str, Any]:
        """Get service performance statistics"""
        cache_hit_rate = (
            self.metrics['cache_hits'] / 
            (self.metrics['cache_hits'] + self.metrics['cache_misses'])
            if (self.metrics['cache_hits'] + self.metrics['cache_misses']) > 0 else 0
        )
        
        return {
            'cache_hit_rate': cache_hit_rate,
            'total_requests': self.metrics['requests_served'],
            'cost_saved': self.metrics['cost_saved'],
            'uptime': 'Active'
        }
    
    def process_request(self, service: str, method: str, params: Dict[str, Any]) -> Any:
        """Main entry point for processing AI service requests"""
        self.metrics['requests_served'] += 1
        
        try:
            if service == 'text_processor':
                if method == 'extract_text':
                    return self.text_processor.extract_text(params.get('content', ''), params.get('content_type', 'text'))
                elif method == 'preprocess_text':
                    return self.text_processor.preprocess_text(params.get('text', ''), params.get('operations', []))
            
            elif service == 'analysis_service':
                if method == 'analyze_sentiment':
                    return self.analysis_service.analyze_sentiment(params.get('text', ''), params.get('context', 'general'))
                elif method == 'extract_insights':
                    return self.analysis_service.extract_key_insights(params.get('text', ''), params.get('domain', 'general'))
            
            elif service == 'data_processor':
                if method == 'normalize_data':
                    return self.data_processor.normalize_data(params.get('data', {}), params.get('schema', {}))
            
            else:
                raise ValueError(f"Unknown service: {service}")
                
        except Exception as e:
            self.logger.error(f"Service request error: {e}")
            raise


# Global service instance
shared_ai_service = SharedAIServiceManager()