"""FastAPI application for Langflow component executor."""

import asyncio
import importlib
import inspect
import json
import logging
import os
import sys
import time
import uuid
from typing import Any

from fastapi import FastAPI, HTTPException
from langchain_core.tools import BaseTool, Tool
from pydantic import BaseModel

# Add lfx to Python path if it exists in the node directory
_node_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_lfx_path = os.path.join(_node_dir, "lfx", "src")
if os.path.exists(_lfx_path) and _lfx_path not in sys.path:
    sys.path.insert(0, _lfx_path)

logger = logging.getLogger(__name__)
if os.path.exists(_lfx_path):
    logger.debug(f"Added lfx to Python path: {_lfx_path}")

# Load component mapping from JSON file
_components_json_path = os.path.join(_node_dir, "components.json")
_component_map: dict[str, str] = {}
print(f"[EXECUTOR] Looking for components.json at: {_components_json_path}")
print(f"[EXECUTOR] Node dir: {_node_dir}")
if os.path.exists(_components_json_path):
    try:
        with open(_components_json_path, "r") as f:
            _component_map = json.load(f)
        print(f"[EXECUTOR] ✅ Loaded {len(_component_map)} component mappings from {_components_json_path}")
        logger.info(f"Loaded {len(_component_map)} component mappings from {_components_json_path}")
    except Exception as e:
        print(f"[EXECUTOR] ❌ Failed to load components.json: {e}")
        logger.warning(f"Failed to load components.json: {e}")
else:
    print(f"[EXECUTOR] ❌ components.json not found at {_components_json_path}")
    logger.warning(f"components.json not found at {_components_json_path}")

app = FastAPI(title="Langflow Executor Node", version="0.1.0")

# Initialize NATS client (lazy connection)
_nats_client = None


async def get_nats_client():
    """Get or create NATS client instance."""
    global _nats_client
    if _nats_client is None:
        logger.info("[NATS] Creating new NATS client instance...")
        from node.nats import NATSClient
        nats_url = os.getenv("NATS_URL", "nats://localhost:4222")
        logger.info(f"[NATS] Connecting to NATS at {nats_url}")
        _nats_client = NATSClient(nats_url=nats_url)
        try:
            await _nats_client.connect()
            logger.info("[NATS] ✅ Successfully connected to NATS")
        except Exception as e:
            logger.warning(f"[NATS] ❌ Failed to connect to NATS (non-critical): {e}", exc_info=True)
            _nats_client = None
    else:
        logger.debug("[NATS] Using existing NATS client instance")
    return _nats_client


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
    attributes: dict[str, Any] | None = None  # Serialized _attributes (loop state, etc.)


class ExecutionRequest(BaseModel):
    """Request to execute a component method."""

    component_state: ComponentState
    method_name: str
    is_async: bool = False
    timeout: int = 30
    message_id: str | None = None  # Unique message ID from backend for tracking published messages


class ExecutionResponse(BaseModel):
    """Response from component execution."""

    result: Any
    success: bool
    result_type: str
    execution_time: float
    error: str | None = None
    message_id: str | None = None  # Unique ID for the published NATS message
    updated_attributes: dict[str, Any] | None = None


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
        print(f"[EXECUTOR] Module path is incorrect ({module_name}), looking up {class_name} in components.json (map size: {len(_component_map)})")
        logger.info(f"Module path is incorrect ({module_name}), looking up correct module for {class_name} in components.json")
        
        # Look up the correct module path from the JSON mapping
        if class_name in _component_map:
            correct_module = _component_map[class_name]
            print(f"[EXECUTOR] ✅ Found mapping: {class_name} -> {correct_module}")
            logger.info(f"Found module mapping: {class_name} -> {correct_module}")
            try:
                module = importlib.import_module(correct_module)
                component_class = getattr(module, class_name)
                print(f"[EXECUTOR] ✅ Successfully loaded {class_name} from {correct_module}")
                logger.info(f"Successfully loaded {class_name} from mapped module {correct_module}")
                return component_class
            except (ImportError, AttributeError) as e:
                print(f"[EXECUTOR] ❌ Failed to load {class_name} from {correct_module}: {e}")
                logger.warning(f"Failed to load {class_name} from mapped module {correct_module}: {e}")
                # Fall back to code execution if module import fails
                if component_code:
                    print(f"[EXECUTOR] Falling back to code execution for {class_name}")
                    logger.info(f"Falling back to code execution for {class_name}")
                    try:
                        return await load_component_from_code(component_code, class_name)
                    except Exception as code_error:
                        logger.error(f"Code execution also failed for {class_name}: {code_error}")
                        # Continue to next fallback attempt
        else:
            print(f"[EXECUTOR] ❌ Component {class_name} not found in components.json (available: {list(_component_map.keys())[:5]}...)")
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
        return value
    
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


def sanitize_tool_inputs(component_params: dict[str, Any], component_class: str | None = None) -> list[BaseTool] | None:
    """Ensure `tools` parameter only contains LangChain tool objects.

    When components (especially agents) run in tool mode, the backend currently
    serializes tool objects into plain dictionaries. Those dictionaries do not
    expose attributes like `.name`, which causes `validate_tool_names` to raise
    AttributeError. Drop any invalid entries so execution can proceed without
    crashing. Workflows that genuinely depend on those tools will still log a
    warning, but at least the agent can run (albeit without tool support).
    """

    tools_value = component_params.get("tools")
    if not tools_value:
        return None

    candidates = tools_value if isinstance(tools_value, list) else [tools_value]
    valid_tools: list[BaseTool] = []
    invalid_types: list[str] = []
    for tool in candidates:
        if isinstance(tool, BaseTool):
            valid_tools.append(tool)
            continue
        reconstructed = reconstruct_tool(tool)
        if reconstructed:
            valid_tools.append(reconstructed)
            continue
        invalid_types.append(type(tool).__name__)

    if invalid_types:
        logger.warning(
            "[%s] Dropping %d invalid tool payload(s); expected LangChain BaseTool instances, got: %s",
            component_class or "Component",
            len(invalid_types),
            ", ".join(sorted(set(invalid_types))),
        )

    component_params["tools"] = valid_tools
    return valid_tools


def reconstruct_tool(value: Any) -> BaseTool | None:
    """Attempt to rebuild a LangChain tool from serialized metadata."""
    if not isinstance(value, dict):
        return None

    name = value.get("name")
    description = value.get("description", "")
    metadata = value.get("metadata", {})
    if not name:
        return None

    def _tool_func(*args, **kwargs):
        logger.warning(
            "Tool '%s' invoked in executor context; returning placeholder response.",
            name,
        )
        return {
            "tool": name,
            "status": "unavailable",
            "message": "Tool cannot execute inside executor context; please route to appropriate node.",
        }

    try:
        # Tool from langchain_core.tools can wrap simple callables
        reconstructed = Tool(
            name=name,
            description=description or metadata.get("display_description", ""),
            func=_tool_func,
            coroutine=None,
        )
        # CRITICAL: Preserve ALL metadata including _component_state for remote tool execution
        # The backend serializes component_state in metadata, and we MUST preserve it
        reconstructed.metadata = metadata.copy() if metadata else {}
        
        # Log if _component_state is present (for debugging)
        if "_component_state" in reconstructed.metadata:
            component_state = reconstructed.metadata["_component_state"]
            params = component_state.get("parameters", {}) if isinstance(component_state, dict) else {}
            logger.info(
                "Reconstructed tool '%s' with _component_state containing %d parameters: %s",
                name,
                len(params),
                list(params.keys()) if params else "NONE",
            )
        else:
            logger.warning(
                "Reconstructed tool '%s' is MISSING _component_state in metadata - remote execution may fail!",
                name,
            )
        
        return reconstructed
    except Exception as exc:
        logger.warning("Failed to reconstruct tool '%s': %s", name, exc)
        return None


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
        import json

        payload_preview = json.dumps(
            {
                "component_class": request.component_state.component_class,
                "method": request.method_name,
                "is_async": request.is_async,
                "timeout": request.timeout,
                "stream_topic": request.component_state.stream_topic,
                "parameters": request.component_state.parameters,
                "input_values": request.component_state.input_values,
            },
            ensure_ascii=False,
            default=str,
        )
    except Exception as preview_err:
        payload_preview = f"<unserializable payload: {preview_err}>"
    print(f"[EXECUTOR] Incoming execution request: {payload_preview}", flush=True)

    try:
        # Log what we received
        stream_topic_value = request.component_state.stream_topic
        log_msg = (
            f"Received execution request: "
            f"class={request.component_state.component_class}, "
            f"module={request.component_state.component_module}, "
            f"code_length={len(request.component_state.component_code or '') if request.component_state.component_code else 0}, "
            f"stream_topic={stream_topic_value}"
        )
        logger.info(log_msg)
        print(f"[EXECUTOR] {log_msg}")  # Also print to ensure visibility
        
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
        deserialized_inputs: dict[str, Any] = {}
        if request.component_state.input_values:
            # Deserialize input values to reconstruct Data/Message objects
            for key, value in request.component_state.input_values.items():
                deserialized = deserialize_input_value(value)
                # Skip None values to avoid overriding defaults with invalid types
                if deserialized is None:
                    logger.debug(
                        "Skipping None input value for %s to preserve component default", key
                    )
                    continue
                deserialized_inputs[key] = deserialized
            component_params.update(deserialized_inputs)
            logger.info(
                f"Merged {len(deserialized_inputs)} input values from upstream components "
                f"(deserialized to proper types)"
            )
        
        if request.component_state.config:
            # Merge config into parameters with _ prefix
            for key, value in request.component_state.config.items():
                component_params[f"_{key}"] = value

        if request.component_state.component_class == "AgentComponent":
            logger.info(
                "[AgentComponent] input keys: %s; tools raw payload: %s",
                list((request.component_state.input_values or {}).keys()),
                (request.component_state.input_values or {}).get("tools"),
            )
            if request.component_state.input_values and request.component_state.input_values.get("tools"):
                sample_tool = request.component_state.input_values["tools"][0]
                logger.debug("[AgentComponent] Sample tool payload keys: %s", list(sample_tool.keys()))
                logger.debug("[AgentComponent] Sample tool metadata: %s", sample_tool.get("metadata"))

        logger.info(
            f"Instantiating {request.component_state.component_class} "
            f"with {len(component_params)} parameters "
            f"(static: {len(request.component_state.parameters)}, "
            f"inputs: {len(request.component_state.input_values or {})}, "
            f"config: {len(request.component_state.config or {})})"
        )
        # Drop None values to mimic Langflow's default handling (unset fields)
        if component_params:
            filtered_params = {
                key: value for key, value in component_params.items() if value is not None
            }
            if len(filtered_params) != len(component_params):
                logger.debug(
                    "Removed %d None-valued parameters before instantiation",
                    len(component_params) - len(filtered_params),
                )
            component_params = filtered_params

        # Ensure `tools` parameter contains valid tool instances only
        sanitized_tools = sanitize_tool_inputs(component_params, request.component_state.component_class)
        if sanitized_tools is not None and "tools" in deserialized_inputs:
            deserialized_inputs["tools"] = sanitized_tools

        component = component_class(**component_params)

        # Restore serialized _attributes/state (e.g., LoopComponent.__loop_state)
        restored_attributes = {}
        if request.component_state.attributes:
            try:
                deserialized_attrs = deserialize_input_value(request.component_state.attributes)
                if isinstance(deserialized_attrs, dict):
                    restored_attributes = deserialized_attrs
            except Exception as attr_err:  # noqa: BLE001
                logger.warning(
                    "Failed to deserialize component attributes for %s: %s",
                    request.component_state.component_class,
                    attr_err,
                )
        if restored_attributes:
            try:
                if not hasattr(component, "_attributes") or component._attributes is None:
                    component._attributes = {}
                component._attributes.update(restored_attributes)
                logger.info(
                    "Restored %d attribute(s) onto %s (keys: %s)",
                    len(restored_attributes),
                    request.component_state.component_class,
                    list(restored_attributes.keys()),
                )
            except Exception as attr_err:  # noqa: BLE001
                logger.warning(
                    "Failed to restore _attributes on %s: %s",
                    request.component_state.component_class,
                    attr_err,
                )

        # Ensure runtime inputs also populate component attributes for template rendering
        if deserialized_inputs:
            try:
                component.set_attributes(deserialized_inputs)
            except Exception as attr_err:
                logger.warning(
                    "Failed to set component attributes from input values (%s): %s",
                    request.component_state.component_class,
                    attr_err,
                )

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

        # Serialize result and capture updated component attributes (if any)
        serialized_result = serialize_result(result)
        updated_attributes = None
        if hasattr(component, "_attributes"):
            try:
                updated_attributes = serialize_result(getattr(component, "_attributes", {}))
            except Exception as attr_err:  # noqa: BLE001
                logger.warning("Failed to serialize component attributes: %s", attr_err)
                updated_attributes = None

        logger.info(
            f"Method {request.method_name} completed successfully "
            f"in {execution_time:.3f}s, result type: {type(result).__name__}"
        )

        # Use message_id from request (generated by backend) or generate one if not provided
        message_id = request.message_id or str(uuid.uuid4())
        
        # Publish result to NATS stream if topic is provided
        if request.component_state.stream_topic:
            topic = request.component_state.stream_topic
            logger.info(f"[NATS] Attempting to publish to topic: {topic} with message_id: {message_id}")
            print(f"[NATS] Attempting to publish to topic: {topic} with message_id: {message_id}")
            try:
                nats_client = await get_nats_client()
                if nats_client:
                    logger.info(f"[NATS] NATS client obtained, preparing publish data...")
                    print(f"[NATS] NATS client obtained, preparing publish data...")
                    # Publish result to NATS with message ID from backend
                    publish_data = {
                        "message_id": message_id,  # Use message_id from backend request
                        "component_id": request.component_state.component_id,
                        "component_class": request.component_state.component_class,
                        "result": serialized_result,
                        "result_type": type(result).__name__,
                        "execution_time": execution_time,
                    }
                    logger.info(f"[NATS] Publishing to topic: {topic}, message_id: {message_id}, data keys: {list(publish_data.keys())}")
                    print(f"[NATS] Publishing to topic: {topic}, message_id: {message_id}, data keys: {list(publish_data.keys())}")
                    # Use the topic directly (already in format: droq.local.public.userid.workflowid.component.out)
                    await nats_client.publish(topic, publish_data)
                    logger.info(f"[NATS] ✅ Successfully published result to NATS topic: {topic} with message_id: {message_id}")
                    print(f"[NATS] ✅ Successfully published result to NATS topic: {topic} with message_id: {message_id}")
                else:
                    logger.warning(f"[NATS] NATS client is None, cannot publish")
                    print(f"[NATS] ⚠️  NATS client is None, cannot publish")
            except Exception as e:
                # Non-critical: log but don't fail execution
                logger.warning(f"[NATS] ❌ Failed to publish to NATS (non-critical): {e}", exc_info=True)
                print(f"[NATS] ❌ Failed to publish to NATS (non-critical): {e}")
        else:
            msg = f"[NATS] ⚠️  No stream_topic provided in request, skipping NATS publish. Component: {request.component_state.component_class}, ID: {request.component_state.component_id}"
            logger.info(msg)
            print(msg)

        return ExecutionResponse(
            result=serialized_result,
            success=True,
            result_type=type(result).__name__,
            execution_time=execution_time,
            message_id=message_id,  # Return message ID (from request or generated) so backend can match it
            updated_attributes=updated_attributes,
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

