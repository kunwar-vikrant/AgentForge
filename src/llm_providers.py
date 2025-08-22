import os
import requests
import json
import time

OPENAI_URL = "https://api.openai.com/v1/chat/completions"
GROK_URL = "https://api.x.ai/v1/chat/completions"
OLLAMA_URL = "http://localhost:11434/api/chat"
OPENAI_MODEL = "gpt-4o-mini"
GROK_MODEL = "grok-beta"
OLLAMA_MODEL = "llama3"

def get_llm_response(provider, prompt, max_retries=3, retry_delay=2):
    """
    Query a supported LLM provider with a prompt and return the response text.
    Supported providers: 'openai', 'grok', 'ollama'.
    Raises ValueError for missing API keys or unsupported providers.
    Raises RuntimeError for API/network errors.
    """
    if not isinstance(provider, str) or not isinstance(prompt, str):
        raise ValueError("Provider and prompt must be strings.")

    provider = provider.lower().strip()
    url, headers, data = None, {}, {}

    if provider == 'openai':
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("Set OPENAI_API_KEY environment variable.")
        url = OPENAI_URL
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        data = {
            "model": OPENAI_MODEL,
            "messages": [{"role": "user", "content": prompt}]
        }
    elif provider == 'grok':
        api_key = os.getenv('XAI_API_KEY')
        if not api_key:
            raise ValueError("Set XAI_API_KEY environment variable.")
        url = GROK_URL
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        data = {
            "model": GROK_MODEL,
            "messages": [{"role": "user", "content": prompt}]
        }
    elif provider == 'ollama':
        url = OLLAMA_URL
        data = {
            "model": OLLAMA_MODEL,
            "messages": [{"role": "user", "content": prompt}]
        }
        headers = {"Content-Type": "application/json"}
    else:
        raise ValueError("Unsupported provider. Use 'openai', 'grok', or 'ollama'.")

    last_exception = None
    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=data, timeout=60)
            response.raise_for_status()
            try:
                resp_json = response.json()
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Failed to decode JSON from {provider}: {e}\nRaw response: {response.text}")

            # OpenAI/Grok: choices[0]['message']['content'], Ollama: choices[0]['message']['content'] or similar
            if 'choices' in resp_json and resp_json['choices']:
                return resp_json['choices'][0]['message']['content']
            # Ollama may return 'message' at top level
            if 'message' in resp_json and isinstance(resp_json['message'], dict):
                return resp_json['message'].get('content', str(resp_json['message']))
            # Unexpected format
            raise RuntimeError(f"Unexpected response format from {provider}: {resp_json}")
        except requests.exceptions.RequestException as e:
            last_exception = e
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                raise RuntimeError(f"Error communicating with {provider} API after {max_retries} attempts: {e}")
        except (KeyError, IndexError, ValueError, RuntimeError) as e:
            raise RuntimeError(f"Error parsing response from {provider}: {e}")