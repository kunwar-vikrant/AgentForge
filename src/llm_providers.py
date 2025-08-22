import os
import requests
import json
import time

import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ProviderConfig:
    """Configuration for an LLM provider."""
    url: str
    model: str
    api_key_env: Optional[str] = None
    default_timeout: int = 120
    max_tokens: Optional[int] = None
    temperature: float = 0.7

# Provider configurations
PROVIDER_CONFIGS = {
    'openai': ProviderConfig(
        url="https://api.openai.com/v1/chat/completions",
        model="gpt-4o-mini",
        api_key_env="OPENAI_API_KEY",
        max_tokens=4000
    ),
    'grok': ProviderConfig(
        url="https://api.x.ai/v1/chat/completions",
        model="grok-beta",
        api_key_env="XAI_API_KEY",
        max_tokens=4000
    ),
    'ollama': ProviderConfig(
        url="http://localhost:11434/api/chat",
        model="llama3",
        max_tokens=4000
    )
}

def validate_provider(provider: str) -> str:
    """Validate and normalize provider name."""
    provider = provider.lower().strip()
    if provider not in PROVIDER_CONFIGS:
        available = ', '.join(PROVIDER_CONFIGS.keys())
        raise ValueError(f"Unsupported provider '{provider}'. Available: {available}")
    return provider

def get_api_key(provider_config: ProviderConfig) -> Optional[str]:
    """Get API key for provider if required."""
    if not provider_config.api_key_env:
        return None
    
    api_key = os.getenv(provider_config.api_key_env)
    if not api_key:
        raise ValueError(f"Set {provider_config.api_key_env} environment variable.")
    return api_key

def build_request_data(provider: str, prompt: str, config: ProviderConfig) -> Dict[str, Any]:
    """Build request data for the specific provider."""
    base_data = {
        "model": config.model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": config.temperature
    }
    
    if config.max_tokens:
        base_data["max_tokens"] = config.max_tokens
    
    # Provider-specific adjustments
    if provider == 'ollama':
        # Ollama might have different parameter names
        base_data["stream"] = False
    
    return base_data

def parse_response(provider: str, response_data: Dict[str, Any]) -> str:
    """Parse response based on provider format."""
    try:
        # Standard OpenAI-compatible format
        if 'choices' in response_data and response_data['choices']:
            content = response_data['choices'][0]['message']['content']
            if content:
                return content.strip()
        
        # Ollama alternative format
        if provider == 'ollama':
            if 'message' in response_data and isinstance(response_data['message'], dict):
                content = response_data['message'].get('content', '')
                if content:
                    return content.strip()
            
            # Sometimes Ollama returns content directly
            if 'content' in response_data:
                return response_data['content'].strip()
        
        # Fallback: convert entire response to string
        logger.warning(f"Unexpected response format from {provider}, using fallback parsing")
        return str(response_data)
        
    except (KeyError, IndexError, TypeError) as e:
        raise RuntimeError(f"Failed to parse response from {provider}: {e}")

def get_llm_response(provider: str, prompt: str, max_retries: int = 2) -> str:
    """
    Query a supported LLM provider with a prompt and return the response text.
    
    Args:
        provider: LLM provider name ('openai', 'grok', 'ollama')
        prompt: Text prompt to send to the LLM
        max_retries: Maximum number of retry attempts for transient failures
    
    Returns:
        Response text from the LLM
        
    Raises:
        ValueError: For invalid provider or missing API keys
        RuntimeError: For API/network errors or parsing failures
    """
    provider = validate_provider(provider)
    config = PROVIDER_CONFIGS[provider]
    
    if not prompt or not prompt.strip():
        raise ValueError("Prompt cannot be empty")
    
    logger.info(f"Querying {provider} with prompt length: {len(prompt)}")
    
    # Build headers
    headers = {"Content-Type": "application/json"}
    api_key = get_api_key(config)
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    
    # Build request data
    data = build_request_data(provider, prompt, config)
    
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            if attempt > 0:
                wait_time = min(2 ** attempt, 10)  # Exponential backoff, max 10s
                logger.info(f"Retrying {provider} request (attempt {attempt + 1}) after {wait_time}s...")
                time.sleep(wait_time)
            
            response = requests.post(
                config.url, 
                headers=headers, 
                json=data, 
                timeout=config.default_timeout
            )
            
            # Log response details for debugging
            logger.debug(f"Response status: {response.status_code}")
            logger.debug(f"Response headers: {dict(response.headers)}")
            
            response.raise_for_status()
            response_data = response.json()
            
            content = parse_response(provider, response_data)
            if not content:
                raise RuntimeError("Received empty response from LLM")
            
            logger.info(f"Successfully received response from {provider} (length: {len(content)})")
            return content
            
        except requests.exceptions.Timeout as e:
            last_error = f"Timeout after {config.default_timeout}s: {e}"
            logger.warning(f"Attempt {attempt + 1} timed out for {provider}")
            
        except requests.exceptions.ConnectionError as e:
            last_error = f"Connection error: {e}"
            logger.warning(f"Attempt {attempt + 1} connection failed for {provider}")
            
        except requests.exceptions.HTTPError as e:
            if response.status_code in [429, 500, 502, 503, 504]:  # Retryable errors
                last_error = f"HTTP {response.status_code}: {e}"
                logger.warning(f"Attempt {attempt + 1} HTTP error {response.status_code} for {provider}")
            else:
                # Non-retryable HTTP errors
                error_detail = ""
                try:
                    error_data = response.json()
                    error_detail = error_data.get('error', {}).get('message', str(error_data))
                except:
                    error_detail = response.text[:200]
                raise RuntimeError(f"HTTP {response.status_code} from {provider}: {error_detail}")
                
        except (json.JSONDecodeError, KeyError, RuntimeError) as e:
            if attempt == max_retries:  # Don't retry parsing errors on last attempt
                raise RuntimeError(f"Response parsing failed for {provider}: {e}")
            last_error = f"Parsing error: {e}"
            logger.warning(f"Attempt {attempt + 1} parsing failed for {provider}")
            
        except Exception as e:
            last_error = f"Unexpected error: {e}"
            logger.warning(f"Attempt {attempt + 1} failed for {provider}: {e}")
    
    # All retries exhausted
    raise RuntimeError(f"Failed to get response from {provider} after {max_retries + 1} attempts. Last error: {last_error}")


