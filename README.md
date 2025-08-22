# AgentForge

A robust Python-based CLI tool that creates a complete agentic workflow for automatically generating AI agents. AgentForge takes a user's description and transforms it into a fully functional AI agent with comprehensive planning, code generation, and testing.

## Features

### 🤖 Multi-LLM Support
- **OpenAI GPT Models** (via API key)
- **Grok** (via xAI API)
- **Ollama** (local models like Llama3)

### 🏗️ Agentic Pipeline
1. **Planning Agent**: Analyzes requirements and creates detailed architecture plans
2. **Code Generation Agent**: Generates production-ready Python code with best practices
3. **Testing Agent**: Creates comprehensive test suites and suggests improvements

### 💪 Robust Features
- **Comprehensive Error Handling**: Retry logic, timeout management, and graceful failure handling
- **Configurable Settings**: JSON-based configuration with environment variable overrides
- **Detailed Logging**: Full audit trail with configurable log levels
- **Organized Output**: Timestamped directories with generated code, tests, and documentation
- **Input Validation**: Thorough validation of inputs and API responses
- **Type Safety**: Full type hints and validation

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd AgentForge
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables for your chosen LLM provider:

### For OpenAI:
```bash
export OPENAI_API_KEY="your-openai-api-key"
```

### For Grok (xAI):
```bash
export XAI_API_KEY="your-xai-api-key"
```

### For Ollama:
Make sure Ollama is running locally:
```bash
ollama serve
```

## Usage

### Basic Usage
```bash
python src/main.py <provider> "<use_case_description>"
```

### Examples

**Create a news summarization agent:**
```bash
python src/main.py openai "Create an agent that summarizes daily news via RSS feeds and emails the summary"
```

**Create a data analysis agent:**
```bash
python src/main.py grok "Build an agent that analyzes CSV data and generates insights with visualizations"
```

**Create a chatbot agent:**
```bash
python src/main.py ollama "Design a customer support chatbot that can handle common questions and escalate complex issues"
```

## Configuration

### Configuration File
Create or modify `config.json` to customize behavior:

```json
{
  "max_retries": 3,
  "min_plan_length": 50,
  "min_code_length": 100,
  "output_base_dir": "generated_agents",
  "create_timestamped_dirs": true,
  "save_logs": true,
  "log_level": "INFO",
  "default_timeout": 120,
  "default_temperature": 0.7,
  "default_max_tokens": 4000,
  "openai_model": "gpt-4o-mini",
  "grok_model": "grok-beta",
  "ollama_model": "llama3"
}
```

### Environment Variables
Override configuration with environment variables:

```bash
export AGENTFORGE_MAX_RETRIES=5
export AGENTFORGE_LOG_LEVEL=DEBUG
export AGENTFORGE_OUTPUT_DIR=my_agents
export OPENAI_MODEL=gpt-4
```

## Output Structure

Each run creates a timestamped directory with:

```
generated_agents/
└── 20240822_143022/
    ├── README.md           # Generation summary
    ├── agent_plan.txt      # Detailed architecture plan
    ├── custom_agent.py     # Generated agent code
    └── test_agent.py       # Test suite
```

## Architecture

### Core Components

- **`main.py`**: Main orchestration logic with robust error handling
- **`llm_providers.py`**: Multi-provider LLM interface with retry logic and proper response parsing
- **`config.py`**: Configuration management with file and environment variable support

### Agentic Workflow

1. **Input Validation**: Validates provider and use case description
2. **Planning Phase**: Creates detailed agent architecture with retry logic
3. **Code Generation**: Produces clean, documented Python code
4. **Testing Phase**: Generates comprehensive test suites
5. **Output Organization**: Saves all artifacts with proper structure

### Error Handling

- **Retry Logic**: Configurable retries for transient failures
- **Timeout Management**: Proper timeout handling for all API calls
- **Graceful Degradation**: Continues operation even if non-critical steps fail
- **Detailed Logging**: Comprehensive logging for debugging and monitoring

## Advanced Features

### Custom Models
Configure different models per provider:
```json
{
  "openai_model": "gpt-4",
  "grok_model": "grok-2",
  "ollama_model": "llama3:8b"
}
```

### Output Customization
Control output behavior:
```json
{
  "create_timestamped_dirs": false,  # Use single output directory
  "output_base_dir": "my_custom_dir",
  "save_logs": false  # Disable log file creation
}
```

### Development Mode
For development, use debug logging:
```bash
export AGENTFORGE_LOG_LEVEL=DEBUG
python src/main.py ollama "test agent"
```

## Dependencies

- **Core**: `requests` for HTTP API calls
- **Development**: `pytest`, `black`, `flake8`, `mypy`
- **Documentation**: `mkdocs`, `mkdocs-material`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes with proper tests
4. Ensure code quality with `black`, `flake8`, and `mypy`
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Troubleshooting

### Common Issues

**API Key not found:**
```bash
# Make sure environment variables are set
echo $OPENAI_API_KEY
echo $XAI_API_KEY
```

**Ollama connection failed:**
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags
```

**Generation timeout:**
```bash
# Increase timeout in config.json
export AGENTFORGE_TIMEOUT=300
```

### Debug Mode
Enable detailed logging:
```bash
export AGENTFORGE_LOG_LEVEL=DEBUG
python src/main.py <provider> "<use_case>"
```

Check the log file for detailed error information:
```bash
tail -f agent_forge.log
```