# Langflow Executor Node

A Droq node that executes Langflow components in isolated environments. This service receives component execution requests from the main Langflow backend and executes them via HTTP API.

## Quick Start

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

# Option 3: Build from node directory (use parent as context)
cd /path/to/droqflow/node
docker build -f Dockerfile -t langflow-executor-node:latest ..

# Run the container
docker run -p 8000:8000 langflow-executor-node:latest
```

## API Endpoints

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

## Environment Variables

- `HOST` - Server host (default: `0.0.0.0`)
- `PORT` - Server port (default: `8000`)
- `LOG_LEVEL` - Logging level (default: `INFO`)
- `RELOAD` - Enable auto-reload for development (default: `false`)

## Architecture

```
Main Langflow Backend
  ↓ HTTP POST /api/v1/execute
Executor Node (this service)
  ↓ Loads component class dynamically
  ↓ Instantiates component
  ↓ Executes method
  ↓ Returns serialized result
Main Langflow Backend
```

## Integration with Main Backend

The main Langflow backend calls this executor node from `Component._get_output_result()` when components need to execute. All components are now routed to the executor by default.

## Docker Build

The Dockerfile is designed to be built from the repository root:

```bash
# From repo root
docker build -f node/Dockerfile -t langflow-executor-node:latest .
```

This allows the Dockerfile to access both:
- `app/src/lfx/` - Langflow source code
- `node/` - Executor node source code

## Next Steps

1. Test locally with sample component execution
2. Build Docker image
3. Deploy executor node service
4. Configure main backend to use executor node

## License

Apache License 2.0
