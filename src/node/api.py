"""FastAPI application for Langflow component executor."""

import asyncio
import importlib
import json
import logging
import os
import sys
import time
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Add lfx to Python path if it exists in the node directory
_node_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_lfx_path = os.path.join(_node_dir, "lfx", "src")
if os.path.exists(_lfx_path) and _lfx_path not in sys.path:
    sys.path.insert(0, _lfx_path)

logger = logging.getLogger(__name__)

# Import NATS client
try:
    from node.nats import NATSClient
except ImportError:
    NATSClient = None
    logger.warning("NATS client not available - publishing to NATS will be disabled")
if os.path.exists(_lfx_path):
    logger.debug(f"Added lfx to Python path: {_lfx_path}")


def _create_precomputed_embeddings(vectors: list, texts: list):
    """Create a LangChain-compatible Embeddings object from pre-computed vectors.
    
    This is a fallback when lfx.components.dfx.embeddings is not available.
    """
    try:
        from langchain.embeddings.base import Embeddings
    except ImportError:
        try:
            from langchain_core.embeddings import Embeddings
        except ImportError:
            # If LangChain isn't available, return a simple wrapper
            class Embeddings:
                pass
    
    class LocalPrecomputedEmbeddings(Embeddings):
        """Local LangChain Embeddings wrapper for pre-computed vectors."""
        
        def __init__(self, vectors: list, texts: list):
            self._vectors = vectors or []
            self._texts = texts or []
            self._text_to_vector = {}
            for i, text in enumerate(self._texts):
                if i < len(self._vectors) and text:
                    self._text_to_vector[text] = self._vectors[i]
        
        def embed_documents(self, texts: list) -> list:
            results = []
            for i, text in enumerate(texts):
                if text in self._text_to_vector:
                    results.append(self._text_to_vector[text])
                elif i < len(self._vectors):
                    results.append(self._vectors[i])
                elif self._vectors:
                    results.append(self._vectors[0])
                else:
                    results.append([])
            return results
        
        def embed_query(self, text: str) -> list:
            if text in self._text_to_vector:
                return self._text_to_vector[text]
            return self._vectors[0] if self._vectors else []
        
        @property
        def vectors(self):
            return self._vectors
        
        @property
        def texts(self):
            return self._texts
    
    return LocalPrecomputedEmbeddings(vectors, texts)


# Load component mapping from JSON file
_components_json_path = os.path.join(_node_dir, "components.json")
_component_map: dict[str, str] = {}
logger.info(f"Looking for components.json at: {_components_json_path}")
logger.info(f"Node dir: {_node_dir}")
if os.path.exists(_components_json_path):
    try:
        with open(_components_json_path, "r") as f:
            _component_map = json.load(f)
        logger.info(f"Loaded {len(_component_map)} component mappings from {_components_json_path}")
    except Exception as e:
        logger.warning(f"Failed to load components.json: {e}")
else:
    logger.warning(f"components.json not found at {_components_json_path}")

app = FastAPI(title="Langflow Executor Node", version="0.1.0")

# Global NATS client instance
_nats_client: NATSClient | None = None


@app.on_event("startup")
async def startup_event():
    """Initialize NATS client on startup."""
    global _nats_client
    if NATSClient is not None:
        try:
            _nats_client = NATSClient()
            await _nats_client.connect()
            logger.info("NATS client initialized and connected")
        except Exception as e:
            logger.warning(f"Failed to initialize NATS client: {e}. Publishing to NATS will be disabled.")
            _nats_client = None
    else:
        logger.warning("NATS client not available - publishing to NATS will be disabled")


@app.on_event("shutdown")
async def shutdown_event():
    """Close NATS connection on shutdown."""
    global _nats_client
    if _nats_client:
        try:
            await _nats_client.close()
            logger.info("NATS connection closed")
        except Exception as e:
            logger.warning(f"Error closing NATS connection: {e}")


class ComponentState(BaseModel):
    """Component state for execution."""

    component_class: str
    component_module: str
    component_code: str | None = None
    parameters: dict[str, Any]
    input_values: dict[str, Any] | None = None  # Current input values from upstream components
    config: dict[str, Any] | None = None
    display_name: str | None = None
    component_id: str | None = None
    stream_topic: str | None = None  # NATS stream topic for publishing results


class ExecutionRequest(BaseModel):
    """Request to execute a component method."""

    component_state: ComponentState
    method_name: str
    is_async: bool = False
    timeout: int = 30
    message_id: str | None = None  # Message ID for NATS publishing


class ExecutionResponse(BaseModel):
    """Response from component execution."""

    result: Any
    success: bool
    result_type: str
    execution_time: float
    error: str | None = None


async def load_component_class(
    module_name: str, class_name: str, component_code: str | None = None
) -> type:
    """
    Dynamically load a component class.

    Args:
        module_name: Python module path (e.g., "lfx.components.input_output.text")
        class_name: Class name (e.g., "TextInputComponent")
        component_code: Optional source code to execute if module loading fails

    Returns:
        Component class

    Raises:
        HTTPException: If module or class cannot be loaded
    """
    # If module path is wrong (validation wrapper), try to find the correct module from components.json
    if module_name in ("lfx.custom.validate", "lfx.custom.custom_component.component"):
        logger.info(f"Module path is incorrect ({module_name}), looking up correct module for {class_name} in components.json")
        
        # Look up the correct module path from the JSON mapping
        if class_name in _component_map:
            correct_module = _component_map[class_name]
            logger.info(f"Found module mapping: {class_name} -> {correct_module}")
            try:
                module = importlib.import_module(correct_module)
                component_class = getattr(module, class_name)
                logger.info(f"Successfully loaded {class_name} from mapped module {correct_module}")
                return component_class
            except (ImportError, AttributeError) as e:
                logger.warning(f"Failed to load {class_name} from mapped module {correct_module}: {e}")
        else:
            logger.warning(f"Component {class_name} not found in components.json mapping")
    
    # First try loading from the provided module path
    try:
        module = importlib.import_module(module_name)
        component_class = getattr(module, class_name)
        logger.info(f"Successfully loaded {class_name} from module {module_name}")
        return component_class
    except ImportError as e:
        logger.warning(f"Failed to import module {module_name}: {e}")
        # If module import fails and we have code, try executing code
        if component_code:
            logger.info(f"Attempting to load {class_name} from provided code")
            return await load_component_from_code(component_code, class_name)
        raise HTTPException(
            status_code=400, detail=f"Failed to import module {module_name}: {e}"
        )
    except AttributeError as e:
        logger.warning(f"Class {class_name} not found in module {module_name}: {e}")
        # If class not found and we have code, try executing code
        if component_code:
            logger.info(
                f"Attempting to load {class_name} from provided code "
                f"(code length: {len(component_code)} chars)"
            )
        else:
            logger.error(
                f"No component_code provided! Cannot fallback to code execution. "
                f"Module={module_name}, Class={class_name}"
            )
        # Try to use code if available
        if component_code:
            try:
                return await load_component_from_code(component_code, class_name)
            except HTTPException as code_error:
                # Provide more context in the error
                logger.error(
                    f"Failed to load from code: {code_error.detail}. "
                    f"Module path was: {module_name}"
                )
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Class {class_name} not found in module {module_name} "
                        f"and code execution failed: {code_error.detail}"
                    ),
                )
        raise HTTPException(
            status_code=400,
            detail=f"Class {class_name} not found in module {module_name}: {e}",
        )


async def load_component_from_code(component_code: str, class_name: str) -> type:
    """
    Load a component class by executing its source code.

    Args:
        component_code: Python source code containing the component class
        class_name: Name of the class to extract

    Returns:
        Component class

    Raises:
        HTTPException: If code execution fails or class not found
    """
    try:
        # Create a new namespace for code execution
        # Import common Langflow modules that components might need
        namespace = {
            "__builtins__": __builtins__,
        }
        
        # Try to import common Langflow modules into the namespace
        try:
            import lfx.base.io.text
            import lfx.io
            import lfx.schema.message
            namespace["lfx"] = __import__("lfx")
            namespace["lfx.base"] = __import__("lfx.base")
            namespace["lfx.base.io"] = __import__("lfx.base.io")
            namespace["lfx.base.io.text"] = lfx.base.io.text
            namespace["lfx.io"] = lfx.io
            namespace["lfx.schema"] = __import__("lfx.schema")
            namespace["lfx.schema.message"] = lfx.schema.message
        except Exception as import_error:
            logger.warning(f"Could not pre-import some modules: {import_error}")
        
        exec(compile(component_code, "<string>", "exec"), namespace)
        
        if class_name not in namespace:
            # Log what classes are available in the namespace
            available_classes = [
                k for k, v in namespace.items()
                if isinstance(v, type) and not k.startswith("_")
            ]
            logger.error(
                f"Class {class_name} not found in provided code. "
                f"Available classes: {available_classes[:10]}"
            )
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Class {class_name} not found in provided code. "
                    f"Available classes: {', '.join(available_classes[:5])}"
                ),
            )
        
        component_class = namespace[class_name]
        logger.info(f"Successfully loaded {class_name} from provided code")
        return component_class
    except SyntaxError as e:
        logger.error(f"Syntax error in component code: {e}")
        raise HTTPException(
            status_code=400, detail=f"Syntax error in component code: {e}"
        )
    except Exception as e:
        logger.error(f"Error executing component code: {e}")
        raise HTTPException(
            status_code=400, detail=f"Error executing component code: {type(e).__name__}: {e}"
        )


def serialize_result(result: Any) -> Any:
    """
    Serialize component execution result.

    Args:
        result: Component execution result

    Returns:
        Serialized result
    """
    # Handle None
    if result is None:
        return None
    
    # Handle primitive types
    if isinstance(result, (str, int, float, bool)):
        return result
    
    # Skip type/metaclass objects - they can't be serialized
    if isinstance(result, type):
        # Return the class name as a string representation
        return f"<class '{result.__module__}.{result.__name__}'>"
    
    # Check for Pydantic metaclass specifically
    result_type_str = str(type(result))
    if "ModelMetaclass" in result_type_str or "metaclass" in result_type_str.lower():
        return f"<metaclass: {getattr(result, '__name__', type(result).__name__)}>"
    
    # Handle DataFrame (langflow wrapper around pandas DataFrame)
    type_name = type(result).__name__
    if type_name == "DataFrame":
        # Check if it has a 'data' attribute that is a pandas DataFrame
        if hasattr(result, "data"):
            inner_data = result.data
            # Check if inner_data is a pandas DataFrame
            if hasattr(inner_data, "to_dict"):
                try:
                    records = inner_data.to_dict(orient="records")
                    return {"type": "DataFrame", "data": records}
                except Exception:
                    pass
        # Maybe it IS a pandas DataFrame directly
        if hasattr(result, "to_dict"):
            try:
                records = result.to_dict(orient="records")
                return {"type": "DataFrame", "data": records}
            except Exception:
                pass
    
    # Handle Data objects (langflow Data type)  
    if type_name == "Data":
        if hasattr(result, "data") and isinstance(result.data, dict):
            return {"type": "Data", "data": result.data}
    
    # Handle lists/tuples first (before other checks)
    if isinstance(result, (list, tuple)):
        return [serialize_result(item) for item in result]
    
    # Handle dicts
    if isinstance(result, dict):
        return {k: serialize_result(v) for k, v in result.items()}
    
    # Handle common Langflow types (Pydantic models)
    if hasattr(result, "model_dump"):
        try:
            dumped = result.model_dump()
            # Recursively serialize the dumped result to catch any nested issues
            return serialize_result(dumped)
        except Exception as e:
            logger.debug(f"model_dump failed: {e}, trying dict()")
            # If model_dump fails, try dict()
            pass
    if hasattr(result, "dict"):
        try:
            dumped = result.dict()
            return serialize_result(dumped)
        except Exception as e:
            logger.debug(f"dict() failed: {e}")
            pass
    
    # Try to serialize via __dict__ (but skip private attributes and classes)
    if hasattr(result, "__dict__"):
        try:
            serialized_dict = {}
            for k, v in result.__dict__.items():
                # Skip private attributes except __class__
                if k.startswith("_") and k != "__class__":
                    continue
                # Skip type objects
                if isinstance(v, type):
                    continue
                serialized_dict[k] = serialize_result(v)
            return serialized_dict
        except Exception as e:
            logger.debug(f"__dict__ serialization failed: {e}")
            pass
    
    # For callable objects (functions, methods), return string representation
    if callable(result):
        return f"<callable: {getattr(result, '__name__', type(result).__name__)}>"
    
    # Last resort: try to convert to string
    try:
        return str(result)
    except Exception:
        return f"<unserializable: {type(result).__name__}>"


def deserialize_input_value(value: Any) -> Any:
    """
    Deserialize input value, reconstructing Langflow types from dicts.
    
    Args:
        value: Serialized input value (may be a dict representing Data/Message)
        
    Returns:
        Deserialized value with proper types reconstructed
    """
    if not isinstance(value, dict):
        # Recursively handle lists
        if isinstance(value, list):
            return [deserialize_input_value(item) for item in value]
        # Log non-dict values for debugging
        if value is not None and value != "":
            logger.debug(f"[DESERIALIZE] Non-dict value: type={type(value).__name__}, value={repr(value)[:100]}")
        return value
    
    # Check if it's a serialized PrecomputedEmbeddings object (from DFX Embeddings component)
    if value.get("type") == "PrecomputedEmbeddings":
        vectors = value.get("vectors", [])
        texts = value.get("texts", [])
        logger.info(f"[DESERIALIZE] 🎯 Found PrecomputedEmbeddings: {len(vectors)} vectors, {len(texts)} texts")
        # Reconstruct PrecomputedEmbeddings from serialized data
        try:
            from lfx.components.dfx.embeddings import PrecomputedEmbeddings
            logger.info(f"[DESERIALIZE] 🎯 Reconstructing PrecomputedEmbeddings from embeddings.py")
            return PrecomputedEmbeddings(vectors=vectors, texts=texts)
        except ImportError:
            # Fallback: try export_embeddings version
            try:
                from lfx.components.dfx.export_embeddings import PrecomputedEmbeddings
                logger.info(f"[DESERIALIZE] 🎯 Reconstructing PrecomputedEmbeddings from export_embeddings.py")
                return PrecomputedEmbeddings(
                    [{"vector": v, "text": t} for v, t in zip(vectors, texts)]
                )
            except ImportError:
                # Last fallback: create a local Embeddings wrapper
                logger.info(f"[DESERIALIZE] 🎯 Creating local PrecomputedEmbeddings wrapper")
                return _create_precomputed_embeddings(vectors, texts)
    
    # Check if it's an Embeddings type marker (fallback serialization)
    if value.get("type") == "Embeddings":
        logger.warning(f"[DESERIALIZE] ⚠️ Found Embeddings type marker without vectors (class: {value.get('class')})")
        return _create_precomputed_embeddings([], [])
    
    # Try to reconstruct Data or Message objects
    try:
        from lfx.schema.message import Message
        from lfx.schema.data import Data
        
        # Check if it looks like a Message (has Message-specific fields)
        # Message extends Data, so it has text_key, data, and Message-specific fields like sender, category, duration, etc.
        message_fields = ["sender", "category", "session_id", "timestamp", "duration", "flow_id", "error", "edit", "sender_name", "context_id"]
        has_message_fields = any(key in value for key in message_fields)
        
        # Also check inside data dict (Message fields might be nested there)
        data_dict = value.get("data", {})
        if isinstance(data_dict, dict):
            has_message_fields_in_data = any(key in data_dict for key in message_fields)
            has_message_fields = has_message_fields or has_message_fields_in_data
        
        if has_message_fields:
            # Fix timestamp format if present (convert various formats to YYYY-MM-DD HH:MM:SS UTC)
            if "timestamp" in value and isinstance(value["timestamp"], str):
                timestamp = value["timestamp"]
                # Convert ISO format with T separator to space (e.g., "2025-11-14T13:09:23 UTC" -> "2025-11-14 13:09:23 UTC")
                if "T" in timestamp:
                    # Replace T with space, but preserve the UTC part
                    timestamp = timestamp.replace("T", " ")
                # Convert ISO format with timezone to UTC format
                if "+00:00" in timestamp:
                    timestamp = timestamp.replace("+00:00", " UTC")
                elif timestamp.endswith("Z"):
                    timestamp = timestamp.replace("Z", " UTC")
                elif "Z " in timestamp:
                    timestamp = timestamp.replace("Z ", " UTC ")
                # Ensure it ends with UTC if it doesn't already
                if not timestamp.endswith(" UTC") and not timestamp.endswith(" UTC"):
                    # Try to parse and reformat using datetime
                    try:
                        from datetime import datetime, timezone
                        # Try common formats
                        for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S %Z"]:
                            try:
                                dt = datetime.strptime(timestamp.strip(), fmt)
                                if dt.tzinfo is None:
                                    dt = dt.replace(tzinfo=timezone.utc)
                                timestamp = dt.strftime("%Y-%m-%d %H:%M:%S %Z")
                                break
                            except ValueError:
                                continue
                    except Exception:
                        pass
                value["timestamp"] = timestamp
            
            # Create Message object - Message constructor will handle merging fields into data dict
            # according to Data.validate_data logic
            try:
                message_obj = Message(**value)
                logger.debug(f"[DESERIALIZE] Successfully reconstructed Message object from dict with keys: {list(value.keys())}")
                return message_obj
            except Exception as msg_error:
                logger.warning(f"[DESERIALIZE] Failed to create Message from dict: {msg_error}, keys: {list(value.keys())}")
                # Try to create with just the data dict if that exists
                if "data" in value and isinstance(value["data"], dict):
                    try:
                        return Message(data=value["data"], **{k: v for k, v in value.items() if k != "data"})
                    except Exception:
                        pass
                raise
        
        # Check if it looks like a Data object (has text_key or data field, but not Message-specific fields)
        if ("data" in value or "text_key" in value) and not has_message_fields:
            return Data(**value)
            
    except Exception as e:
        logger.debug(f"[DESERIALIZE] Could not reconstruct object from dict: {e}")
        # Return as-is if reconstruction fails
        pass
    
    # For dicts, recursively deserialize values
    return {k: deserialize_input_value(v) for k, v in value.items()}


@app.post("/api/v1/execute", response_model=ExecutionResponse)
async def execute_component(request: ExecutionRequest) -> ExecutionResponse:
    """
    Execute a Langflow component method.

    Args:
        request: Execution request with component state and method name

    Returns:
        Execution response with result or error
    """
    start_time = time.time()

    try:
        # Log what we received
        logger.info(
            f"Received execution request: "
            f"class={request.component_state.component_class}, "
            f"module={request.component_state.component_module}, "
            f"code_length={len(request.component_state.component_code or '') if request.component_state.component_code else 0}"
        )
        
        # Load component class dynamically
        component_class = await load_component_class(
            request.component_state.component_module,
            request.component_state.component_class,
            request.component_state.component_code,
        )

        # Instantiate component with parameters
        component_params = request.component_state.parameters.copy()
        
        # Merge input_values (runtime values from upstream components) into parameters
        # These override static parameters since they contain the actual workflow data
        if request.component_state.input_values:
            # Log raw input values for debugging
            for k, v in request.component_state.input_values.items():
                v_type = type(v).__name__
                if isinstance(v, dict):
                    v_preview = f"dict with keys: {list(v.keys())[:5]}, type={v.get('type')}"
                elif isinstance(v, str):
                    v_preview = f"str len={len(v)}, preview={repr(v[:100])}"
                elif isinstance(v, list):
                    v_preview = f"list len={len(v)}"
                else:
                    v_preview = repr(v)[:100]
                logger.info(f"[RAW INPUT] {k}: {v_type} = {v_preview}")
            
            # Deserialize input values to reconstruct Data/Message objects
            deserialized_inputs = {
                k: deserialize_input_value(v) 
                for k, v in request.component_state.input_values.items()
            }
            
            # Log deserialized values
            for k, v in deserialized_inputs.items():
                logger.info(f"[DESERIALIZED INPUT] {k}: {type(v).__name__}")
            
            component_params.update(deserialized_inputs)
            logger.info(
                f"Merged {len(request.component_state.input_values)} input values from upstream components "
                f"(deserialized to proper types)"
            )
        
        if request.component_state.config:
            # Merge config into parameters with _ prefix
            for key, value in request.component_state.config.items():
                component_params[f"_{key}"] = value

        logger.info(
            f"Instantiating {request.component_state.component_class} "
            f"with {len(component_params)} parameters "
            f"(static: {len(request.component_state.parameters)}, "
            f"inputs: {len(request.component_state.input_values or {})}, "
            f"config: {len(request.component_state.config or {})})"
        )
        component = component_class(**component_params)

        # Get the method
        if not hasattr(component, request.method_name):
            raise HTTPException(
                status_code=400,
                detail=f"Method {request.method_name} not found on component",
            )

        method = getattr(component, request.method_name)

        # Execute method (handle async/sync)
        logger.info(
            f"Executing method {request.method_name} "
            f"(async={request.is_async}) on {request.component_state.component_class}"
        )

        if request.is_async:
            result = await asyncio.wait_for(method(), timeout=request.timeout)
        else:
            # Run sync method in thread pool
            result = await asyncio.wait_for(
                asyncio.to_thread(method), timeout=request.timeout
            )

        execution_time = time.time() - start_time

        # Serialize result
        serialized_result = serialize_result(result)

        # Check if stream_topic is provided (backend expects NATS publishing)
        stream_topic = request.component_state.stream_topic
        message_id = request.message_id
        
        # Publish to NATS if stream_topic is provided
        if stream_topic and _nats_client and _nats_client.js:
            try:
                # Prepare message payload matching backend's expected format
                nats_payload = {
                    "message_id": message_id,
                    "result": serialized_result,
                    "result_type": type(result).__name__,
                    "execution_time": execution_time,
                    "success": True,
                }
                
                # Publish directly to the full stream_topic using JetStream
                # stream_topic format: droq.local.public.{user_id}.{flow_id}.{component_id}.out
                # We publish directly without adding stream_name prefix (it's already in the topic)
                payload_bytes = json.dumps(nats_payload).encode()
                headers = {"message_id": message_id} if message_id else None
                
                await _nats_client.js.publish(stream_topic, payload_bytes, headers=headers)
                logger.info(
                    f"[NATS] ✅ Published result to stream_topic={stream_topic}, message_id={message_id}"
                )
            except Exception as e:
                logger.error(
                    f"[NATS] ❌ Failed to publish to NATS: {e}. "
                    f"Backend will fall back to HTTP result after timeout.",
                    exc_info=True
                )
        elif stream_topic and not _nats_client:
            logger.warning(
                f"[NATS] ⚠️ stream_topic={stream_topic} provided but NATS client not available. "
                f"Backend will wait ~5+ seconds for NATS timeout before using HTTP result."
            )
        else:
            logger.debug(
                f"[NATS] No stream_topic provided - HTTP-only response (no NATS publishing needed)"
            )

        logger.info(
            f"Method {request.method_name} completed successfully "
            f"in {execution_time:.3f}s, result type: {type(result).__name__}"
        )

        return ExecutionResponse(
            result=serialized_result,
            success=True,
            result_type=type(result).__name__,
            execution_time=execution_time,
        )

    except asyncio.TimeoutError:
        execution_time = time.time() - start_time
        error_msg = f"Execution timed out after {request.timeout}s"
        logger.error(error_msg)
        return ExecutionResponse(
            result=None,
            success=False,
            result_type="TimeoutError",
            execution_time=execution_time,
            error=error_msg,
        )

    except HTTPException:
        raise

    except Exception as e:
        execution_time = time.time() - start_time
        error_msg = f"Execution failed: {type(e).__name__}: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return ExecutionResponse(
            result=None,
            success=False,
            result_type=type(e).__name__,
            execution_time=execution_time,
            error=error_msg,
        )


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy", "service": "langflow-executor-node"}


@app.get("/")
async def root() -> dict[str, Any]:
    """Root endpoint."""
    return {
        "service": "Langflow Executor Node",
        "version": "0.1.0",
        "endpoints": {
            "execute": "/api/v1/execute",
            "health": "/health",
        },
    }

