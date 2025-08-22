import os
import requests
import json

def get_llm_response(provider, prompt):
    """
    Query a supported LLM provider with a prompt and return the response text.
    Supported providers: 'openai', 'grok', 'ollama'.
    Raises ValueError for missing API keys or unsupported providers.
    Raises RuntimeError for API/network errors.
    """
    provider = provider.lower().strip()
    url, headers, data = None, {}, {}

    if provider == 'openai':
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("Set OPENAI_API_KEY environment variable.")
        url = "https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        data = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": prompt}]
        }
    elif provider == 'grok':
        api_key = os.getenv('XAI_API_KEY')
        if not api_key:
            raise ValueError("Set XAI_API_KEY environment variable.")
        url = "https://api.x.ai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        data = {
            "model": "grok-beta",
            "messages": [{"role": "user", "content": prompt}]
        }
    elif provider == 'ollama':
        url = "http://localhost:11434/api/chat"
        data = {
            "model": "llama3",
            "messages": [{"role": "user", "content": prompt}]
        }
        headers = {"Content-Type": "application/json"}
    else:
        raise ValueError("Unsupported provider. Use 'openai', 'grok', or 'ollama'.")

    try:
        response = requests.post(url, headers=headers, json=data, timeout=60)
        response.raise_for_status()
        resp_json = response.json()
        # OpenAI/Grok: choices[0]['message']['content'], Ollama: choices[0]['message']['content'] or similar
        if 'choices' in resp_json and resp_json['choices']:
            return resp_json['choices'][0]['message']['content']
        # Ollama may return 'message' at top level
        if 'message' in resp_json and isinstance(resp_json['message'], dict):
            return resp_json['message'].get('content', str(resp_json['message']))
        return str(resp_json)
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Error communicating with {provider} API: {e}")
    except (KeyError, IndexError, ValueError) as e:
        raise RuntimeError(f"Unexpected response format from {provider}: {e}\nRaw response: {response.text}")