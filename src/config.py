"""
Configuration management for AgentForge.
"""
import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict

@dataclass
class AgentForgeConfig:
    """Main configuration for AgentForge."""
    # Retry settings
    max_retries: int = 3
    min_plan_length: int = 50
    min_code_length: int = 100
    
    # Output settings
    output_base_dir: str = "generated_agents"
    create_timestamped_dirs: bool = True
    save_logs: bool = True
    log_level: str = "INFO"
    
    # LLM settings
    default_timeout: int = 120
    default_temperature: float = 0.7
    default_max_tokens: int = 4000
    
    # Provider-specific settings
    openai_model: str = "gpt-4o-mini"
    grok_model: str = "grok-beta"
    ollama_model: str = "llama3"
    ollama_host: str = "localhost"
    ollama_port: int = 11434
    
    @classmethod
    def load_from_file(cls, config_path: Optional[Path] = None) -> 'AgentForgeConfig':
        """Load configuration from file, falling back to defaults."""
        if config_path is None:
            config_path = Path("config.json")
        
        if config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                return cls(**config_data)
            except (json.JSONDecodeError, TypeError) as e:
                print(f"Warning: Failed to load config from {config_path}: {e}")
                print("Using default configuration.")
        
        return cls()
    
    def save_to_file(self, config_path: Optional[Path] = None) -> None:
        """Save current configuration to file."""
        if config_path is None:
            config_path = Path("config.json")
        
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(asdict(self), f, indent=2)
    
    @classmethod
    def load_from_env(cls) -> 'AgentForgeConfig':
        """Load configuration from environment variables."""
        config = cls()
        
        # Override with environment variables if present
        env_mappings = {
            'AGENTFORGE_MAX_RETRIES': ('max_retries', int),
            'AGENTFORGE_MIN_PLAN_LENGTH': ('min_plan_length', int),
            'AGENTFORGE_MIN_CODE_LENGTH': ('min_code_length', int),
            'AGENTFORGE_OUTPUT_DIR': ('output_base_dir', str),
            'AGENTFORGE_LOG_LEVEL': ('log_level', str),
            'AGENTFORGE_TIMEOUT': ('default_timeout', int),
            'AGENTFORGE_TEMPERATURE': ('default_temperature', float),
            'AGENTFORGE_MAX_TOKENS': ('default_max_tokens', int),
            'OPENAI_MODEL': ('openai_model', str),
            'GROK_MODEL': ('grok_model', str),
            'OLLAMA_MODEL': ('ollama_model', str),
            'OLLAMA_HOST': ('ollama_host', str),
            'OLLAMA_PORT': ('ollama_port', int),
        }
        
        for env_var, (attr_name, type_func) in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                try:
                    setattr(config, attr_name, type_func(env_value))
                except (ValueError, TypeError) as e:
                    print(f"Warning: Invalid value for {env_var}: {env_value} ({e})")
        
        return config

# Global configuration instance
_config: Optional[AgentForgeConfig] = None

def get_config() -> AgentForgeConfig:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        # Try to load from file first, then from environment
        _config = AgentForgeConfig.load_from_file()
        
        # Override with environment variables
        env_config = AgentForgeConfig.load_from_env()
        for key, value in asdict(env_config).items():
            if os.getenv(f'AGENTFORGE_{key.upper()}') is not None:
                setattr(_config, key, value)
    
    return _config

def set_config(config: AgentForgeConfig) -> None:
    """Set the global configuration instance."""
    global _config
    _config = config
