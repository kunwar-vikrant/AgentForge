import os
import requests
import json

def get_llm_response(provider, prompt):
    if provider == 'openai':
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("Set OPENAI_API_KEY environment variable.")
        url = "https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        data = {
            "model": "gpt-4o-mini",  # Or preferred model
            "messages": [{"role": "user", "content": prompt}]
        }
    elif provider == 'grok':
        api_key = os.getenv('XAI_API_KEY')
        if not api_key:
            raise ValueError("Set XAI_API_KEY environment variable.")
        url = "https://api.x.ai/v1/chat/completions"  # Assuming xAI's API endpoint
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        data = {
            "model": "grok-beta",  # Or current Grok model
            "messages": [{"role": "user", "content": prompt}]
        }
    elif provider == 'ollama':
        url = "http://localhost:11434/api/chat"  # Ollama local API
        data = {
            "model": "llama3",  # Or installed model
            "messages": [{"role": "user", "content": prompt}]
        }
        headers = {"Content-Type": "application/json"}
    else:
        raise ValueError("Unsupported provider. Use 'openai', 'grok', or 'ollama'.")
    
    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    return response.json()['choices'][0]['message']['content']