# DroqFlow Langflow Executor Node

**DroqFlow Langflow Executor Node** provides a unified interface for building, deploying, and managing Langflow workflows — simplifying workflow automation and lifecycle operations with comprehensive AI model integrations and component orchestration.

## 🚀 Installation

### Using UV (Recommended)

```bash
# Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create and install DroqFlow Langflow Executor
uv init my-droqflow-project && cd my-droqflow-project
uv add "droqflow-langflow-executor @ git+ssh://git@github.com/droq-ai/lfx-runtime-executor-node.git@main"

# Verify installation
uv run executor-node --help
```

## 🧩 Usage

```python
import droqflow

yaml_content = """
workflow:
  name: langflow-executor-workflow
  version: "1.0.0"
  description: A workflow for Langflow component execution

  nodes:
    - name: langflow-executor
      type: executor
      did: did:droq:node:langflow-executor-v1
      output: streams.droq.langflow-executor.local.public.executor.out
      source_code:
        path: "./src"
        type: "local"
        docker:
          type: "file"
          dockerfile: "./Dockerfile"
      config:
        host: "0.0.0.0"
        port: 8000
        log_level: "INFO"
        locality: "local"
        remote_endpoint: "nats://droq-nats-server:4222"

  streams:
    sources:
      - streams.droq.langflow-executor.local.public.executor.out

permissions: []
"""

builder = droqflow.DroqWorkflowBuilder(yaml_content=yaml_content)
builder.load_workflow()
builder.generate_artifacts(output_dir="artifacts")
```

## 🔧 Configuration

The executor node can be configured using environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `HOST` | `0.0.0.0` | Server host address |
| `PORT` | `8000` | Server port |
| `LOG_LEVEL` | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR) |
| `RELOAD` | `false` | Enable auto-reload for development |

### Docker Configuration

```yaml
config:
  description: "Langflow Executor Node with comprehensive AI model integrations"
  docker_image: "droq/langflow-executor:v1"
  host: "0.0.0.0"
  port: 8000
  log_level: "INFO"
  locality: "local"
  remote_endpoint: "nats://droq-nats-server:4222"
```

## 📦 Component Categories

### AI Model Providers
- **Anthropic** - Generate text using Anthropic's Messages API and models
- **OpenAI** - Generate text using OpenAI models
- **Google Generative AI** - Google's generative AI models
- **Azure OpenAI** - Microsoft Azure OpenAI service
- **NVIDIA** - Generates text using NVIDIA LLMs
- **Hugging Face** - Hugging Face model endpoints
- **Cohere** - Cohere language models
- **Mistral** - Mistral AI models
- **Ollama** - Local Ollama models
- And many more...

### Vector Stores
- **FAISS** - FAISS Vector Store with search capabilities
- **Chroma** - Chroma vector database
- **Pinecone** - Pinecone vector database
- **Weaviate** - Weaviate vector database
- **Redis** - Redis vector store
- And many more...

### Data Processing
- **API Request** - Make HTTP requests using URL or cURL commands
- **Filter Data** - Filter data based on specified criteria
- **Combine Text** - Concatenate text sources using delimiters
- **Type Converter** - Convert between different data types
- **DataFrame Operations** - Operations on pandas DataFrames

### Logic & Flow Control
- **If-Else (Conditional Router)** - Route messages based on text comparison
- **Smart Router (LLM Conditional Router)** - Route messages using LLM categorization
- **Loop** - Create loops in flow execution
- **Run Flow** - Execute sub-flows within main flows

### Integrations
- **GitHub** - GitHub repository operations
- **Slack** - Slack messaging integration
- **Google Calendar** - Google Calendar operations
- **Gmail** - Gmail email operations
- **Notion** - Notion database and page operations
- And many more...

## 🚀 Quick Start

### Local Development

**Recommended: Use the startup script**

```bash
cd node
./start-local.sh
```

This script will:
- Check for dependencies
- Install Langflow if needed
- Start the executor node with auto-reload enabled

**Manual startup:**

```bash
# 1. Install dependencies (if needed)
cd node
uv sync

# 2. Install Langflow dependencies (if needed)
uv pip install -e ../app/src/lfx

# 3. Run the executor
PYTHONPATH=src python -m node.main
# or with auto-reload (default in start-local.sh)
RELOAD=true PYTHONPATH=src python -m node.main
```

The executor will run on `http://localhost:8000` by default.

### Docker Build

**Important:** The Dockerfile must be built from the repository root, not from the `node/` directory.

```bash
# Option 1: Use the build script (recommended)
cd /path/to/droqflow
./node/build.sh

# Option 2: Build manually from repo root
cd /path/to/droqflow
docker build -f node/Dockerfile -t langflow-executor-node:latest .

# Run the container
docker run -p 8000:8000 langflow-executor-node:latest
```

## 🌐 API Endpoints

### POST /api/v1/execute

Execute a Langflow component method.

**Request:**
```json
{
    "component_state": {
        "component_class": "TextInput",
        "component_module": "lfx.components.input_output.text_input",
        "parameters": {
            "input_value": "Hello World"
        },
        "config": {},
        "display_name": "Text Input",
        "component_id": "TextInput-abc123"
    },
    "method_name": "text_response",
    "is_async": false,
    "timeout": 30
}
```

**Response:**
```json
{
    "result": {...},
    "success": true,
    "result_type": "Message",
    "execution_time": 0.123,
    "error": null
}
```

### GET /health

Health check endpoint.

### GET /

Service information.

## 🏗️ Architecture

```
Main Langflow Backend
  ↓ HTTP POST /api/v1/execute
Langflow Executor Node (this service)
  ↓ Loads component class dynamically
  ↓ Instantiates component
  ↓ Executes method
  ↓ Returns serialized result
Main Langflow Backend
```

## 🧪 Development

### Running Tests

```bash
# Install development dependencies
uv sync --dev

# Run tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=lfx --cov-report=html
```

### Code Quality

```bash
# Format code
uv run black .

# Lint code
uv run ruff check .

# Type checking
uv run mypy lfx/
```

## 📚 Documentation

* [Installation Guide](docs/installation.md)
* [Usage Guide](docs/usage.md)
* [API Reference](docs/api.md)
* [Component Development](docs/component-development.md)
* [Docker Deployment](docs/docker-deployment.md)

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

* [Documentation](https://github.com/droq-ai/lfx-runtime-executor-node#readme)
* [Issue Tracker](https://github.com/droq-ai/lfx-runtime-executor-node/issues)
* [Discord Community](https://discord.gg/droqai)
* [Email Support](mailto:support@droq.ai)