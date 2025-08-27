# AgentForge

Create single agents or robust multi-agent systems from plain English. One command plans, generates, quality‑checks, and packages a runnable agent; another runs a supervisor + workers loop.

## Features

### 🤖 Multi-LLM Support
- **OpenAI GPT Models** (via API key)
- **Grok** (via xAI API)
- **Groq** 
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

## 🚀 Quick Start

```bash
git clone <repository-url>
cd AgentForge
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e '.[dev]'

# Single agent (plan + code + tests + packaging)
agentforge generate --provider openai --use-case "Summarize daily sales CSVs and flag anomalies"

# Multi-agent (offline deterministic)
agentforge multi --task "Explain caching layers" --provider echo --verbose

# Planning only
agentforge plan --provider ollama --use-case "Design an FAQ chatbot"
```

Set environment variables for real providers:

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

## 🧩 Single-Agent Generation (Detailed)

The `generate` subcommand performs:
1. Planning (LLM architecture plan with retries)
2. Code generation + heuristic quality evaluation & refinement
3. Test suggestions
4. Deterministic fallback template if quality fails
5. Packaging (requirements, run scripts, Dockerfile, README)

Artifacts land in `generated_agents/<timestamp>/`.

Legacy direct call (still works):
```bash
python src/main.py <provider> "<use case>"
```

### Run as an API Service
Start the FastAPI server (after installing new dependencies):
```bash
uvicorn src.api.app:app --reload
```
Then call endpoints:
```bash
curl -X POST http://127.0.0.1:8000/plan -H 'Content-Type: application/json' \
  -d '{"provider":"openai","use_case":"Build an agent that summarizes emails"}'
```
Full pipeline:
```bash
curl -X POST http://127.0.0.1:8000/pipeline -H 'Content-Type: application/json' \
  -d '{"provider":"ollama","use_case":"Create an agent that tags support tickets"}'
```

### More Examples
```bash
agentforge generate --provider grok --use-case "Analyze CSV data and output anomaly report"
agentforge generate --provider ollama --use-case "Customer support chatbot that escalates complex issues"
agentforge generate --provider openai --use-case "News summarizer that emails an AM briefing"
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

## 🤝 Multi-Agent Orchestration

Run a supervisor + worker loop:
```bash
agentforge multi --task "Draft phased migration plan" --provider openai
agentforge multi --task "Summarize caching strategy" --provider echo --verbose
```
Supervisor must output `NEXT:<agent>` or `FINISH:<answer>`.
Workers output `RESPOND:<answer>` or `TOOL:<name>:<arg>`.

Add an extra worker (snippet):
```python
from agents.base import BaseAgent
from agents.adapters import EchoLLM
from agents.orchestrator import Orchestrator
llm = EchoLLM()
sup = BaseAgent("supervisor", llm, "Supervisor: decide.")
worker = BaseAgent("worker", llm, "Worker: solve tasks.")
researcher = BaseAgent("researcher", llm, "Research facts.")
orch = Orchestrator({"worker": worker, "researcher": researcher}, sup, max_turns=8)
```
Offline deterministic path (no keys): `--provider echo`.

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

## 🛠️ Troubleshooting

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
## 📚 Extended Documentation

See `docs/USAGE.md` for advanced multi-agent usage, tool authoring, roadmap, and a troubleshooting matrix.
```bash
tail -f agent_forge.log
```
