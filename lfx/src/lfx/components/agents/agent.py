import asyncio
import json
import os
import re
import uuid
from copy import deepcopy
from typing import Any

import httpx
from langchain_core.tools import StructuredTool, Tool, ToolException
from pydantic import ValidationError

from lfx.base.agents.agent import LCToolsAgentComponent
from lfx.base.agents.events import ExceptionWithMessageError
from lfx.base.models.model_input_constants import (
    ALL_PROVIDER_FIELDS,
    MODEL_DYNAMIC_UPDATE_FIELDS,
    MODEL_PROVIDERS_DICT,
    MODEL_PROVIDERS_LIST,
    MODELS_METADATA,
)
from lfx.base.models.model_utils import get_model_name
from lfx.components.helpers.current_date import CurrentDateComponent
from lfx.components.helpers.memory import MemoryComponent
from lfx.components.langchain_utilities.tool_calling import ToolCallingAgentComponent
from lfx.custom.custom_component.component import get_component_toolkit, get_local_node_id
from lfx.custom.utils import update_component_build_config
from lfx.helpers.base_model import build_model_from_schema
from lfx.inputs.inputs import BoolInput
from lfx.io import DropdownInput, IntInput, MessageTextInput, MultilineInput, Output, TableInput
from lfx.log.logger import logger
from lfx.schema.data import Data
from lfx.schema.dotdict import dotdict
from lfx.schema.message import Message
from lfx.schema.table import EditMode


def set_advanced_true(component_input):
    component_input.advanced = True
    return component_input


class AgentComponent(ToolCallingAgentComponent):
    display_name: str = "Agent"
    description: str = "Define the agent's instructions, then enter a task to complete using tools."
    documentation: str = "https://docs.langflow.org/agents"
    icon = "bot"
    beta = False
    name = "Agent"

    memory_inputs = [set_advanced_true(component_input) for component_input in MemoryComponent().inputs]

    # Filter out json_mode from OpenAI inputs since we handle structured output differently
    if "OpenAI" in MODEL_PROVIDERS_DICT:
        openai_inputs_filtered = [
            input_field
            for input_field in MODEL_PROVIDERS_DICT["OpenAI"]["inputs"]
            if not (hasattr(input_field, "name") and input_field.name == "json_mode")
        ]
    else:
        openai_inputs_filtered = []

    inputs = [
        DropdownInput(
            name="agent_llm",
            display_name="Model Provider",
            info="The provider of the language model that the agent will use to generate responses.",
            options=[*MODEL_PROVIDERS_LIST],
            value="OpenAI",
            real_time_refresh=True,
            refresh_button=False,
            input_types=[],
            options_metadata=[MODELS_METADATA[key] for key in MODEL_PROVIDERS_LIST if key in MODELS_METADATA],
            external_options={
                "fields": {
                    "data": {
                        "node": {
                            "name": "connect_other_models",
                            "display_name": "Connect other models",
                            "icon": "CornerDownLeft",
                        }
                    }
                },
            },
        ),
        *openai_inputs_filtered,
        MultilineInput(
            name="system_prompt",
            display_name="Agent Instructions",
            info="System Prompt: Initial instructions and context provided to guide the agent's behavior.",
            value="You are a helpful assistant that can use tools to answer questions and perform tasks.",
            advanced=False,
        ),
        MessageTextInput(
            name="context_id",
            display_name="Context ID",
            info="The context ID of the chat. Adds an extra layer to the local memory.",
            value="",
            advanced=True,
        ),
        IntInput(
            name="n_messages",
            display_name="Number of Chat History Messages",
            value=100,
            info="Number of chat history messages to retrieve.",
            advanced=True,
            show=True,
        ),
        MultilineInput(
            name="format_instructions",
            display_name="Output Format Instructions",
            info="Generic Template for structured output formatting. Valid only with Structured response.",
            value=(
                "You are an AI that extracts structured JSON objects from unstructured text. "
                "Use a predefined schema with expected types (str, int, float, bool, dict). "
                "Extract ALL relevant instances that match the schema - if multiple patterns exist, capture them all. "
                "Fill missing or ambiguous values with defaults: null for missing values. "
                "Remove exact duplicates but keep variations that have different field values. "
                "Always return valid JSON in the expected format, never throw errors. "
                "If multiple objects can be extracted, return them all in the structured format."
            ),
            advanced=True,
        ),
        TableInput(
            name="output_schema",
            display_name="Output Schema",
            info=(
                "Schema Validation: Define the structure and data types for structured output. "
                "No validation if no output schema."
            ),
            advanced=True,
            required=False,
            value=[],
            table_schema=[
                {
                    "name": "name",
                    "display_name": "Name",
                    "type": "str",
                    "description": "Specify the name of the output field.",
                    "default": "field",
                    "edit_mode": EditMode.INLINE,
                },
                {
                    "name": "description",
                    "display_name": "Description",
                    "type": "str",
                    "description": "Describe the purpose of the output field.",
                    "default": "description of field",
                    "edit_mode": EditMode.POPOVER,
                },
                {
                    "name": "type",
                    "display_name": "Type",
                    "type": "str",
                    "edit_mode": EditMode.INLINE,
                    "description": ("Indicate the data type of the output field (e.g., str, int, float, bool, dict)."),
                    "options": ["str", "int", "float", "bool", "dict"],
                    "default": "str",
                },
                {
                    "name": "multiple",
                    "display_name": "As List",
                    "type": "boolean",
                    "description": "Set to True if this output field should be a list of the specified type.",
                    "default": "False",
                    "edit_mode": EditMode.INLINE,
                },
            ],
        ),
        *LCToolsAgentComponent.get_base_inputs(),
        # removed memory inputs from agent component
        # *memory_inputs,
        BoolInput(
            name="add_current_date_tool",
            display_name="Current Date",
            advanced=True,
            info="If true, will add a tool to the agent that returns the current date.",
            value=True,
        ),
    ]
    outputs = [
        Output(name="response", display_name="Response", method="message_response"),
    ]

    async def get_agent_requirements(self):
        """Get the agent requirements for the agent."""
        print("[AGENT] 🚀 get_agent_requirements CALLED", flush=True)
        logger.info("🚀 get_agent_requirements CALLED")
        llm_model, display_name = await self.get_llm()
        if llm_model is None:
            msg = "No language model selected. Please choose a model to proceed."
            raise ValueError(msg)
        self.model_name = get_model_name(llm_model, display_name=display_name)

        # Get memory data
        self.chat_history = await self.get_memory_data()
        await logger.adebug(f"Retrieved {len(self.chat_history)} chat history messages")
        if isinstance(self.chat_history, Message):
            self.chat_history = [self.chat_history]

        # Add current date tool if enabled
        if self.add_current_date_tool:
            if not isinstance(self.tools, list):  # type: ignore[has-type]
                self.tools = []
            current_date_tool = (await CurrentDateComponent(**self.get_base_args()).to_toolkit()).pop(0)
            if not isinstance(current_date_tool, StructuredTool):
                msg = "CurrentDateComponent must be converted to a StructuredTool"
                raise TypeError(msg)
            self.tools.append(current_date_tool)
        
        print(f"[AGENT] 🔧 About to call _prepare_remote_tools with {len(self.tools) if self.tools else 0} tool(s)", flush=True)
        logger.info("🔧 About to call _prepare_remote_tools with tools=%s", self.tools)
        logger.info("🔧 Tools count: %d, Tools type: %s", len(self.tools) if self.tools else 0, type(self.tools))
        self.tools = self._prepare_remote_tools(self.tools or [])
        print(f"[AGENT] ✅ _prepare_remote_tools returned, final tools count: {len(self.tools) if self.tools else 0}", flush=True)
        logger.info("✅ _prepare_remote_tools returned, final tools count: %d", len(self.tools) if self.tools else 0)
        return llm_model, self.chat_history, self.tools

    async def message_response(self) -> Message:
        try:
            print("[AGENT] 🎯 message_response CALLED - About to get agent requirements", flush=True)
            logger.info("🎯 message_response CALLED - About to get agent requirements")
            llm_model, self.chat_history, self.tools = await self.get_agent_requirements()
            logger.info("🎯 Got agent requirements - tools count: %d", len(self.tools) if self.tools else 0)
            # Set up and run agent
            self.set(
                llm=llm_model,
                tools=self.tools or [],
                chat_history=self.chat_history,
                input_value=self.input_value,
                system_prompt=self.system_prompt,
            )
            logger.info("🎯 Creating agent runnable with %d tools", len(self.tools) if self.tools else 0)
            agent = self.create_agent_runnable()
            logger.info("🎯 Running agent...")
            result = await self.run_agent(agent)

            # Store result for potential JSON output
            self._agent_result = result

        except (ValueError, TypeError, KeyError) as e:
            await logger.aerror(f"{type(e).__name__}: {e!s}")
            raise
        except ExceptionWithMessageError as e:
            await logger.aerror(f"ExceptionWithMessageError occurred: {e}")
            raise
        # Avoid catching blind Exception; let truly unexpected exceptions propagate
        except Exception as e:
            await logger.aerror(f"Unexpected error: {e!s}")
            raise
        else:
            return result

    def _preprocess_schema(self, schema):
        """Preprocess schema to ensure correct data types for build_model_from_schema."""
        processed_schema = []
        for field in schema:
            processed_field = {
                "name": str(field.get("name", "field")),
                "type": str(field.get("type", "str")),
                "description": str(field.get("description", "")),
                "multiple": field.get("multiple", False),
            }
            # Ensure multiple is handled correctly
            if isinstance(processed_field["multiple"], str):
                processed_field["multiple"] = processed_field["multiple"].lower() in [
                    "true",
                    "1",
                    "t",
                    "y",
                    "yes",
                ]
            processed_schema.append(processed_field)
        return processed_schema

    async def build_structured_output_base(self, content: str):
        """Build structured output with optional BaseModel validation."""
        json_pattern = r"\{.*\}"
        schema_error_msg = "Try setting an output schema"

        # Try to parse content as JSON first
        json_data = None
        try:
            json_data = json.loads(content)
        except json.JSONDecodeError:
            json_match = re.search(json_pattern, content, re.DOTALL)
            if json_match:
                try:
                    json_data = json.loads(json_match.group())
                except json.JSONDecodeError:
                    return {"content": content, "error": schema_error_msg}
            else:
                return {"content": content, "error": schema_error_msg}

        # If no output schema provided, return parsed JSON without validation
        if not hasattr(self, "output_schema") or not self.output_schema or len(self.output_schema) == 0:
            return json_data

        # Use BaseModel validation with schema
        try:
            processed_schema = self._preprocess_schema(self.output_schema)
            output_model = build_model_from_schema(processed_schema)

            # Validate against the schema
            if isinstance(json_data, list):
                # Multiple objects
                validated_objects = []
                for item in json_data:
                    try:
                        validated_obj = output_model.model_validate(item)
                        validated_objects.append(validated_obj.model_dump())
                    except ValidationError as e:
                        await logger.aerror(f"Validation error for item: {e}")
                        # Include invalid items with error info
                        validated_objects.append({"data": item, "validation_error": str(e)})
                return validated_objects

            # Single object
            try:
                validated_obj = output_model.model_validate(json_data)
                return [validated_obj.model_dump()]  # Return as list for consistency
            except ValidationError as e:
                await logger.aerror(f"Validation error: {e}")
                return [{"data": json_data, "validation_error": str(e)}]

        except (TypeError, ValueError) as e:
            await logger.aerror(f"Error building structured output: {e}")
            # Fallback to parsed JSON without validation
            return json_data

    async def json_response(self) -> Data:
        """Convert agent response to structured JSON Data output with schema validation."""
        # Always use structured chat agent for JSON response mode for better JSON formatting
        try:
            system_components = []

            # 1. Agent Instructions (system_prompt)
            agent_instructions = getattr(self, "system_prompt", "") or ""
            if agent_instructions:
                system_components.append(f"{agent_instructions}")

            # 2. Format Instructions
            format_instructions = getattr(self, "format_instructions", "") or ""
            if format_instructions:
                system_components.append(f"Format instructions: {format_instructions}")

            # 3. Schema Information from BaseModel
            if hasattr(self, "output_schema") and self.output_schema and len(self.output_schema) > 0:
                try:
                    processed_schema = self._preprocess_schema(self.output_schema)
                    output_model = build_model_from_schema(processed_schema)
                    schema_dict = output_model.model_json_schema()
                    schema_info = (
                        "You are given some text that may include format instructions, "
                        "explanations, or other content alongside a JSON schema.\n\n"
                        "Your task:\n"
                        "- Extract only the JSON schema.\n"
                        "- Return it as valid JSON.\n"
                        "- Do not include format instructions, explanations, or extra text.\n\n"
                        "Input:\n"
                        f"{json.dumps(schema_dict, indent=2)}\n\n"
                        "Output (only JSON schema):"
                    )
                    system_components.append(schema_info)
                except (ValidationError, ValueError, TypeError, KeyError) as e:
                    await logger.aerror(f"Could not build schema for prompt: {e}", exc_info=True)

            # Combine all components
            combined_instructions = "\n\n".join(system_components) if system_components else ""
            llm_model, self.chat_history, self.tools = await self.get_agent_requirements()
            self.set(
                llm=llm_model,
                tools=self.tools or [],
                chat_history=self.chat_history,
                input_value=self.input_value,
                system_prompt=combined_instructions,
            )

            # Create and run structured chat agent
            try:
                structured_agent = self.create_agent_runnable()
            except (NotImplementedError, ValueError, TypeError) as e:
                await logger.aerror(f"Error with structured chat agent: {e}")
                raise
            try:
                result = await self.run_agent(structured_agent)
            except (
                ExceptionWithMessageError,
                ValueError,
                TypeError,
                RuntimeError,
            ) as e:
                await logger.aerror(f"Error with structured agent result: {e}")
                raise
            # Extract content from structured agent result
            if hasattr(result, "content"):
                content = result.content
            elif hasattr(result, "text"):
                content = result.text
            else:
                content = str(result)

        except (
            ExceptionWithMessageError,
            ValueError,
            TypeError,
            NotImplementedError,
            AttributeError,
        ) as e:
            await logger.aerror(f"Error with structured chat agent: {e}")
            # Fallback to regular agent
            content_str = "No content returned from agent"
            return Data(data={"content": content_str, "error": str(e)})

        # Process with structured output validation
        try:
            structured_output = await self.build_structured_output_base(content)

            # Handle different output formats
            if isinstance(structured_output, list) and structured_output:
                if len(structured_output) == 1:
                    return Data(data=structured_output[0])
                return Data(data={"results": structured_output})
            if isinstance(structured_output, dict):
                return Data(data=structured_output)
            return Data(data={"content": content})

        except (ValueError, TypeError) as e:
            await logger.aerror(f"Error in structured output processing: {e}")
            return Data(data={"content": content, "error": str(e)})

    async def get_memory_data(self):
        # TODO: This is a temporary fix to avoid message duplication. We should develop a function for this.
        messages = (
            await MemoryComponent(**self.get_base_args())
            .set(
                session_id=self.graph.session_id,
                context_id=self.context_id,
                order="Ascending",
                n_messages=self.n_messages,
            )
            .retrieve_messages()
        )
        return [
            message for message in messages if getattr(message, "id", None) != getattr(self.input_value, "id", None)
        ]

    async def get_llm(self):
        if not isinstance(self.agent_llm, str):
            return self.agent_llm, None

        try:
            provider_info = MODEL_PROVIDERS_DICT.get(self.agent_llm)
            if not provider_info:
                msg = f"Invalid model provider: {self.agent_llm}"
                raise ValueError(msg)

            component_class = provider_info.get("component_class")
            display_name = component_class.display_name
            inputs = provider_info.get("inputs")
            prefix = provider_info.get("prefix", "")

            return self._build_llm_model(component_class, inputs, prefix), display_name

        except (AttributeError, ValueError, TypeError, RuntimeError) as e:
            await logger.aerror(f"Error building {self.agent_llm} language model: {e!s}")
            msg = f"Failed to initialize language model: {e!s}"
            raise ValueError(msg) from e

    def _build_llm_model(self, component, inputs, prefix=""):
        model_kwargs = {}
        for input_ in inputs:
            if hasattr(self, f"{prefix}{input_.name}"):
                model_kwargs[input_.name] = getattr(self, f"{prefix}{input_.name}")
        return component.set(**model_kwargs).build_model()

    def set_component_params(self, component):
        provider_info = MODEL_PROVIDERS_DICT.get(self.agent_llm)
        if provider_info:
            inputs = provider_info.get("inputs")
            prefix = provider_info.get("prefix")
            # Filter out json_mode and only use attributes that exist on this component
            model_kwargs = {}
            for input_ in inputs:
                if hasattr(self, f"{prefix}{input_.name}"):
                    model_kwargs[input_.name] = getattr(self, f"{prefix}{input_.name}")

            return component.set(**model_kwargs)
        return component

    def delete_fields(self, build_config: dotdict, fields: dict | list[str]) -> None:
        """Delete specified fields from build_config."""
        for field in fields:
            build_config.pop(field, None)

    def update_input_types(self, build_config: dotdict) -> dotdict:
        """Update input types for all fields in build_config."""
        for key, value in build_config.items():
            if isinstance(value, dict):
                if value.get("input_types") is None:
                    build_config[key]["input_types"] = []
            elif hasattr(value, "input_types") and value.input_types is None:
                value.input_types = []
        return build_config

    async def update_build_config(
        self, build_config: dotdict, field_value: str, field_name: str | None = None
    ) -> dotdict:
        # Iterate over all providers in the MODEL_PROVIDERS_DICT
        # Existing logic for updating build_config
        if field_name in ("agent_llm",):
            build_config["agent_llm"]["value"] = field_value
            provider_info = MODEL_PROVIDERS_DICT.get(field_value)
            if provider_info:
                component_class = provider_info.get("component_class")
                if component_class and hasattr(component_class, "update_build_config"):
                    # Call the component class's update_build_config method
                    build_config = await update_component_build_config(
                        component_class, build_config, field_value, "model_name"
                    )

            provider_configs: dict[str, tuple[dict, list[dict]]] = {
                provider: (
                    MODEL_PROVIDERS_DICT[provider]["fields"],
                    [
                        MODEL_PROVIDERS_DICT[other_provider]["fields"]
                        for other_provider in MODEL_PROVIDERS_DICT
                        if other_provider != provider
                    ],
                )
                for provider in MODEL_PROVIDERS_DICT
            }
            if field_value in provider_configs:
                fields_to_add, fields_to_delete = provider_configs[field_value]

                # Delete fields from other providers
                for fields in fields_to_delete:
                    self.delete_fields(build_config, fields)

                # Add provider-specific fields
                if field_value == "OpenAI" and not any(field in build_config for field in fields_to_add):
                    build_config.update(fields_to_add)
                else:
                    build_config.update(fields_to_add)
                # Reset input types for agent_llm
                build_config["agent_llm"]["input_types"] = []
                build_config["agent_llm"]["display_name"] = "Model Provider"
            elif field_value == "connect_other_models":
                # Delete all provider fields
                self.delete_fields(build_config, ALL_PROVIDER_FIELDS)
                # # Update with custom component
                custom_component = DropdownInput(
                    name="agent_llm",
                    display_name="Language Model",
                    info="The provider of the language model that the agent will use to generate responses.",
                    options=[*MODEL_PROVIDERS_LIST],
                    real_time_refresh=True,
                    refresh_button=False,
                    input_types=["LanguageModel"],
                    placeholder="Awaiting model input.",
                    options_metadata=[MODELS_METADATA[key] for key in MODEL_PROVIDERS_LIST if key in MODELS_METADATA],
                    external_options={
                        "fields": {
                            "data": {
                                "node": {
                                    "name": "connect_other_models",
                                    "display_name": "Connect other models",
                                    "icon": "CornerDownLeft",
                                },
                            }
                        },
                    },
                )
                build_config.update({"agent_llm": custom_component.to_dict()})
            # Update input types for all fields
            build_config = self.update_input_types(build_config)

            # Validate required keys
            default_keys = [
                "code",
                "_type",
                "agent_llm",
                "tools",
                "input_value",
                "add_current_date_tool",
                "system_prompt",
                "agent_description",
                "max_iterations",
                "handle_parsing_errors",
                "verbose",
            ]
            missing_keys = [key for key in default_keys if key not in build_config]
            if missing_keys:
                msg = f"Missing required keys in build_config: {missing_keys}"
                raise ValueError(msg)
        if (
            isinstance(self.agent_llm, str)
            and self.agent_llm in MODEL_PROVIDERS_DICT
            and field_name in MODEL_DYNAMIC_UPDATE_FIELDS
        ):
            provider_info = MODEL_PROVIDERS_DICT.get(self.agent_llm)
            if provider_info:
                component_class = provider_info.get("component_class")
                component_class = self.set_component_params(component_class)
                prefix = provider_info.get("prefix")
                if component_class and hasattr(component_class, "update_build_config"):
                    # Call each component class's update_build_config method
                    # remove the prefix from the field_name
                    if isinstance(field_name, str) and isinstance(prefix, str):
                        field_name = field_name.replace(prefix, "")
                    build_config = await update_component_build_config(
                        component_class, build_config, field_value, "model_name"
                    )
        return dotdict({k: v.to_dict() if hasattr(v, "to_dict") else v for k, v in build_config.items()})

    async def _get_tools(self) -> list[Tool]:
        component_toolkit = get_component_toolkit()
        tools_names = self._build_tools_names()
        agent_description = self.get_tool_description()
        # TODO: Agent Description Depreciated Feature to be removed
        description = f"{agent_description}{tools_names}"
        tools = component_toolkit(component=self).get_tools(
            tool_name="Call_Agent",
            tool_description=description,
            callbacks=self.get_langchain_callbacks(),
        )
        if hasattr(self, "tools_metadata"):
            tools = component_toolkit(component=self, metadata=self.tools_metadata).update_tools_metadata(tools=tools)
        return tools

    def _prepare_remote_tools(self, tools: list[Tool]) -> list[Tool]:
        print(f"[AGENT] 🔍 _prepare_remote_tools CALLED with {len(tools) if tools else 0} tool(s)", flush=True)
        logger.info("🔍 _prepare_remote_tools CALLED with %d tool(s)", len(tools) if tools else 0)
        if not tools:
            logger.info("_prepare_remote_tools: No tools to process")
            return tools

        logger.info(
            "_prepare_remote_tools: Processing %d tool(s) for remote routing",
            len(tools),
        )
        for tool in tools:
            metadata = getattr(tool, "metadata", {}) or {}
            tool_name = getattr(tool, "name", "<unknown>")
            metadata_keys = sorted(metadata.keys()) if metadata else []
            logger.info(
                "_prepare_remote_tools: tool='%s' metadata_keys=%s metadata=%s",
                tool_name,
                metadata_keys,
                metadata,
            )
        wrapped: list[Tool] = []
        for tool in tools:
            wrapped.append(self._maybe_wrap_remote_tool(tool))
        remote_count = sum(1 for t in wrapped if getattr(t, "metadata", {}).get("_remote_proxy"))
        if remote_count > 0:
            logger.info(
                "_prepare_remote_tools: Wrapped %d/%d tool(s) for remote execution",
                remote_count,
                len(wrapped),
            )
        else:
            logger.warning(
                "_prepare_remote_tools: NO tools were wrapped for remote execution (all %d tools will execute locally)",
                len(wrapped),
            )
        return wrapped

    def _maybe_wrap_remote_tool(self, tool: Tool) -> Tool:
        metadata = getattr(tool, "metadata", {}) or {}
        tool_name = getattr(tool, "name", "<unknown>")
        
        print(f"[AGENT] _maybe_wrap_remote_tool: tool='{tool_name}', metadata_keys={sorted(metadata.keys()) if metadata else []}", flush=True)
        
        if metadata.get("_remote_proxy"):
            print(f"[AGENT] Tool '{tool_name}' already has _remote_proxy flag, skipping", flush=True)
            return tool

        # For plain Tool objects, try to look up component from tool name or convert to StructuredTool
        if not isinstance(tool, StructuredTool):
            print(f"[AGENT] Tool '{tool_name}' is not a StructuredTool (type={type(tool).__name__}), attempting to recover metadata...", flush=True)
            
            # Try to find component_class from metadata or by looking up tool name
            component_class = metadata.get("component_class")
            
            # If still no component_class, try to infer from tool name
            # Common patterns: build_output -> AgentQL, get_current_date -> CurrentDateComponent
            if not component_class:
                # Try to infer component class from tool name
                # This is a fallback - ideally metadata should be preserved during serialization
                tool_to_component_map = {
                    "build_output": "AgentQL",  # AgentQL component's output method
                    "get_current_date": "CurrentDateComponent",
                }
                component_class = tool_to_component_map.get(tool_name)
                if component_class:
                    print(f"[AGENT] Inferred component_class='{component_class}' from tool name '{tool_name}'", flush=True)
                    metadata["component_class"] = component_class
                else:
                    print(f"[AGENT] Tool '{tool_name}' missing component_class; cannot route without component info", flush=True)
                    logger.debug(
                        "Tool '%s' is a plain Tool without component_class metadata; cannot route via registry.",
                        tool.name,
                    )
                    return tool
            
            # If we found component_class, convert to StructuredTool to proceed
            # We'll create args_schema later after we get executor_meta
            args_schema = getattr(tool, "args_schema", None)
            
            tool = StructuredTool(
                name=tool.name,
                description=tool.description,
                func=getattr(tool, "func", None),
                coroutine=getattr(tool, "coroutine", None),
                args_schema=args_schema,
                handle_tool_error=getattr(tool, "handle_tool_error", True),
                callbacks=getattr(tool, "callbacks", None),
                tags=getattr(tool, "tags", None),
                metadata=metadata,
            )
            print(f"[AGENT] Converted plain Tool '{tool_name}' to StructuredTool with component_class='{component_class}'", flush=True)

        # Query registry automatically using component_class from metadata
        component_class = metadata.get("component_class")
        if not component_class:
            print(f"[AGENT] Tool '{tool_name}' missing component_class metadata; cannot route via registry", flush=True)
            logger.debug(
                "Tool '%s' missing component_class metadata; cannot route via registry.",
                tool.name,
            )
            return tool

        print(f"[AGENT] Tool '{tool_name}' has component_class='{component_class}', querying registry...", flush=True)
        from lfx.custom.custom_component.component import _fetch_executor_node_metadata
        executor_meta = _fetch_executor_node_metadata(component_class)
        if not executor_meta:
            print(f"[AGENT] Registry returned no executor for component '{component_class}'; executing locally", flush=True)
            logger.debug(
                "Registry returned no executor for component '%s'; executing locally.",
                component_class,
            )
            return tool

        node_id = executor_meta.get("node_id")
        local_node_id = get_local_node_id()
        print(f"[AGENT] Registry returned executor_meta: node_id='{node_id}', local_node_id='{local_node_id}'", flush=True)
        
        if not node_id or node_id == local_node_id:
            print(f"[AGENT] Tool '{tool_name}' should execute locally (node_id={node_id}, local={local_node_id})", flush=True)
            return tool

        print(
            f"[AGENT] Registry routed tool '{tool_name}' (component={component_class}) to node '{node_id}'",
            flush=True,
        )
        logger.info(
            "Registry routed tool '%s' (component=%s) to node '%s'",
            tool.name,
            component_class,
            node_id,
        )

        # Store executor metadata in tool metadata for later use
        metadata["executor_node"] = executor_meta

        method_name = metadata.get("_component_method")
        component_state = metadata.get("_component_state")
        
        # If missing, try to reconstruct from available information
        if method_name is None:
            # The tool name is usually the method name
            method_name = tool_name
            metadata["_component_method"] = method_name
            print(f"[AGENT] Reconstructed method_name='{method_name}' from tool name", flush=True)
        
        if component_state is None:
            # Reconstruct minimal component_state from available information
            # Get component_module from registry (it returns module_path)
            component_module = executor_meta.get("module_path")
            if not component_module:
                # Fallback: try to look up in components.json
                import json
                import os
                components_json_path = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
                    "components.json"
                )
                if os.path.exists(components_json_path):
                    try:
                        with open(components_json_path, "r") as f:
                            components_map = json.load(f)
                            component_module = components_map.get(component_class)
                    except Exception:
                        pass
                
                # Final fallback: construct from component_class
                if not component_module:
                    # Special cases for known components
                    component_module_map = {
                        "AgentQL": "lfx.components.agentql.agentql_api",
                    }
                    component_module = component_module_map.get(component_class, f"lfx.components.{component_class.lower()}")
            
            # Get stream topic from current component (agent)
            stream_topic = None
            if hasattr(self, "_build_stream_topic"):
                try:
                    stream_topic = self._build_stream_topic()
                except Exception:
                    pass
            
            if not stream_topic:
                # Build stream topic manually if _build_stream_topic doesn't work
                user_id = getattr(self, "user_id", None) or getattr(self, "_user_id", None)
                flow_id = None
                if hasattr(self, "graph") and self.graph:
                    flow_id = getattr(self.graph, "flow_id", None)
                elif hasattr(self, "_flow_id"):
                    flow_id = self._flow_id
                
                if user_id and flow_id:
                    component_id = metadata.get("component_id") or f"{component_class}-{tool_name}"
                    stream_topic = f"droq.local.public.{user_id}.{flow_id}.{component_id}.out"
            
            if not stream_topic:
                # Final fallback: generate a unique topic for this remote invocation
                stream_topic = f"droq.remote.tools.{uuid.uuid4()}.{tool_name}.out"
            
            # Try to get component parameters from the graph if available
            # This is a best-effort attempt - in executor node we may not have full graph access
            parameters = {}
            if hasattr(self, "graph") and self.graph and hasattr(self.graph, "vertices"):
                try:
                    # Try to find the component in the graph by component_id or component_class
                    component_id_to_find = metadata.get("component_id")
                    for vertex in self.graph.vertices:
                        if hasattr(vertex, "custom_component") and vertex.custom_component:
                            comp = vertex.custom_component
                            if (component_id_to_find and comp.get_id() == component_id_to_find) or \
                               (comp.__class__.__name__ == component_class):
                                # Get the component's parameters
                                if hasattr(comp, "_parameters"):
                                    parameters = comp._parameters.copy() if comp._parameters else {}
                                elif hasattr(comp, "get_parameters"):
                                    parameters = comp.get_parameters() or {}
                                print(f"[AGENT] Found component in graph, extracted {len(parameters)} parameters", flush=True)
                                break
                except Exception as exc:
                    print(f"[AGENT] Could not extract parameters from graph: {exc}", flush=True)
                    logger.debug(f"Could not extract component parameters from graph: {exc}")
            
            # Use component_state from metadata if available (includes parameters from backend)
            component_state = metadata.get("_component_state")
            if component_state:
                # Update with any missing fields, but preserve existing parameters
                component_state.setdefault("component_class", component_class)
                component_state.setdefault("component_module", component_module)
                component_state.setdefault("component_id", metadata.get("component_id") or f"{component_class}-{tool_name}")
                component_state.setdefault("stream_topic", stream_topic)
                component_state.setdefault("input_values", {})
                # Merge any parameters we found from graph (but don't overwrite backend parameters)
                if parameters:
                    existing_params = component_state.get("parameters", {})
                    component_state["parameters"] = {**existing_params, **parameters}
                print(f"[AGENT] Using component_state from metadata for '{tool_name}' (includes parameters from backend): {list(component_state.get('parameters', {}).keys())}", flush=True)
            else:
                # Fallback: build minimal component_state (shouldn't happen if backend serialized correctly)
                component_state = {
                    "component_class": component_class,
                    "component_module": component_module,
                    "component_id": metadata.get("component_id") or f"{component_class}-{tool_name}",
                    "stream_topic": stream_topic,
                    "parameters": parameters,  # Include any parameters we found
                    "input_values": {},  # Will be populated from tool args when invoked
                }
                metadata["_component_state"] = component_state
                print(f"[AGENT] WARNING: Reconstructed component_state for '{tool_name}' (missing from metadata - backend should have included it!)", flush=True)
            
            metadata["_component_is_async"] = False  # Default to sync, can be updated if needed
        
        print(f"[AGENT] Tool '{tool_name}': method_name={method_name}, component_state={'present' if component_state else 'MISSING'}", flush=True)

        # Try to create args_schema if missing (needed so agent knows what arguments to pass)
        args_schema = getattr(tool, "args_schema", None)
        if not args_schema:
            try:
                # Dynamically load component class and create args_schema from tool_mode inputs
                component_module = executor_meta.get("module_path") or component_state.get("component_module")
                if component_module:
                    # Import the module - component_module is the full module path like "lfx.components.agentql.agentql_api"
                    # The class name is in component_class (e.g., "AgentQL")
                    module_parts = component_module.split(".")
                    module_name = ".".join(module_parts[:-1]) if len(module_parts) > 1 else component_module
                    # Try to import the module
                    component_mod = __import__(component_module, fromlist=[component_class])
                    # Get the class from the module using component_class name
                    component_cls = getattr(component_mod, component_class)
                    
                    if not isinstance(component_cls, type):
                        # If that didn't work, try getting from the last part of module path
                        last_part = module_parts[-1]
                        if hasattr(component_mod, last_part):
                            potential_cls = getattr(component_mod, last_part)
                            if isinstance(potential_cls, type):
                                component_cls = potential_cls
                    
                    # Create a temporary instance to get tool_mode inputs and extract default parameters
                    base_args = self.get_base_args() if hasattr(self, "get_base_args") else {}
                    temp_component = component_cls(**base_args)
                    tool_mode_inputs = [inp for inp in temp_component.inputs if getattr(inp, "tool_mode", False)]
                    
                    # Only extract default parameters if component_state doesn't already have parameters from backend
                    # The backend should have serialized all parameters including api_key
                    existing_params = component_state.get("parameters", {}) if component_state else {}
                    if not existing_params:
                        print(f"[AGENT] WARNING: component_state has no parameters for '{tool_name}' - backend should have included them!", flush=True)
                        # Fallback: try to extract from component inputs (but this shouldn't be necessary)
                        default_parameters = {}
                        for inp in temp_component.inputs:
                            if not getattr(inp, "tool_mode", False):
                                param_value = None
                                # Try to get from component's current value
                                if hasattr(temp_component, inp.name):
                                    param_value = getattr(temp_component, inp.name)
                                
                                # If still None, try default value
                                if param_value is None:
                                    if hasattr(inp, "value") and inp.value is not None:
                                        param_value = inp.value
                                    elif hasattr(inp, "default") and inp.default is not None:
                                        param_value = inp.default
                                
                                if param_value is not None:
                                    default_parameters[inp.name] = param_value
                        
                        # Update component_state with default parameters if we found any
                        if default_parameters and component_state:
                            component_state["parameters"] = default_parameters
                            metadata["_component_state"] = component_state
                            print(f"[AGENT] Extracted {len(default_parameters)} fallback parameters: {list(default_parameters.keys())}", flush=True)
                    else:
                        print(f"[AGENT] Using parameters from backend for '{tool_name}': {list(existing_params.keys())}", flush=True)
                    
                    if tool_mode_inputs:
                        from lfx.io.schema import create_input_schema
                        args_schema = create_input_schema(tool_mode_inputs)
                        print(f"[AGENT] Created args_schema for '{tool_name}' with {len(tool_mode_inputs)} tool_mode inputs: {[inp.name for inp in tool_mode_inputs]}", flush=True)
            except Exception as exc:
                print(f"[AGENT] Failed to create args_schema for '{tool_name}': {exc}", flush=True)
                import traceback
                print(f"[AGENT] Traceback: {traceback.format_exc()}", flush=True)
                logger.debug(f"Failed to create args_schema for tool '{tool_name}': {exc}", exc_info=True)

        print(f"[AGENT] ✅ Wrapping tool '{tool_name}' for remote execution to node '{node_id}'", flush=True)
        remote_coroutine = self._build_remote_tool_coroutine(
            tool=tool,
            metadata=metadata,
        )
        metadata["_remote_proxy"] = True

        return StructuredTool(
            name=tool.name,
            description=tool.description,
            coroutine=remote_coroutine,
            args_schema=args_schema,
            handle_tool_error=getattr(tool, "handle_tool_error", True),
            callbacks=getattr(tool, "callbacks", None),
            tags=getattr(tool, "tags", None),
            metadata=metadata,
        )

    def _build_remote_tool_coroutine(
        self,
        *,
        tool: StructuredTool,
        metadata: dict[str, Any],
    ):
        async def _remote_tool_runner(*args, **kwargs):
            return await self._invoke_remote_tool(
                tool_metadata=metadata,
                args=args,
                kwargs=kwargs,
                tool_name=tool.name,
            )

        return _remote_tool_runner

    async def _invoke_remote_tool(
        self,
        *,
        tool_metadata: dict[str, Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        tool_name: str,
    ) -> Any:
        executor_meta = tool_metadata.get("executor_node") or {}
        api_url = executor_meta.get("api_url")
        if not api_url:
            raise ToolException(
                f"Remote tool '{tool_name}' is missing an executor api_url."
            )

        method_name = tool_metadata.get("_component_method")
        if not method_name:
            raise ToolException(
                f"Remote tool '{tool_name}' is missing target method metadata."
            )

        is_async = bool(tool_metadata.get("_component_is_async"))

        component_state = deepcopy(tool_metadata.get("_component_state") or {})
        if not component_state:
            raise ToolException(
                f"Remote tool '{tool_name}' is missing serialized component state."
            )

        # Log what parameters we have before merging runtime inputs
        params = component_state.get("parameters", {})
        print(f"[AGENT] _invoke_remote_tool: component_state has {len(params)} parameters: {list(params.keys())}", flush=True)
        if not params:
            print(f"[AGENT] WARNING: component_state['parameters'] is EMPTY for '{tool_name}' - this will cause API key errors!", flush=True)
            print(f"[AGENT] Full component_state keys: {list(component_state.keys())}", flush=True)
        else:
            # Log parameter values - SHOW ACTUAL API KEY FOR AgentQL
            param_preview = {}
            for key, value in params.items():
                if "key" in key.lower() or "secret" in key.lower() or "password" in key.lower():
                    # PRINT ACTUAL VALUE FOR AgentQL API KEY
                    if component_state.get("component_class") == "AgentQL" and key == "api_key":
                        print(f"[AGENT] 🎯 AgentQL API KEY in component_state['parameters']: {repr(value)}", flush=True)
                        param_preview[key] = f"VALUE={repr(value)}" if value else "MISSING/None"
                    else:
                        param_preview[key] = "***" if value else "MISSING/None"
                else:
                    param_preview[key] = str(value)[:50] if value is not None else "None"
            print(f"[AGENT] Parameter values preview: {param_preview}", flush=True)

        stream_topic = component_state.get("stream_topic")
        if not stream_topic:
            raise ToolException(
                f"Remote tool '{tool_name}' is missing stream topic metadata."
            )

        runtime_inputs = self._coerce_tool_inputs(args, kwargs, tool_metadata)
        base_inputs = component_state.get("input_values") or {}
        merged_inputs = {**base_inputs, **runtime_inputs}
        component_state["input_values"] = merged_inputs or None
        
        # Log final state before sending - SHOW AgentQL API KEY
        final_params = component_state.get("parameters", {})
        print(f"[AGENT] Sending to remote executor: parameters={list(final_params.keys())}, input_values={list(merged_inputs.keys())}", flush=True)
        if component_state.get("component_class") == "AgentQL" and "api_key" in final_params:
            api_key_val = final_params.get("api_key")
            print(f"[AGENT] 🎯 FINAL AgentQL API KEY being sent to remote executor: {repr(api_key_val)}", flush=True)

        timeout = int(os.getenv("REMOTE_TOOL_EXECUTION_TIMEOUT", "60"))
        message_id = str(uuid.uuid4())
        payload = {
            "component_state": component_state,
            "method_name": method_name,
            "is_async": is_async,
            "timeout": timeout,
            "message_id": message_id,
        }

        try:
            print(
                f"[AGENT] Proxying tool '{tool_name}' to executor node '{executor_meta.get('node_id') or api_url}' (method={method_name}, async={is_async})",
                flush=True,
            )
            logger.info(
                "Proxying tool '%s' to executor node '%s' (method=%s, async=%s)",
                tool_name,
                executor_meta.get("node_id") or api_url,
                method_name,
                is_async,
            )
            async with httpx.AsyncClient(timeout=timeout + 10) as client:
                response = await client.post(
                    f"{api_url.rstrip('/')}/api/v1/execute",
                    json=payload,
                )
                response.raise_for_status()
                response_data = response.json()
        except httpx.HTTPError as exc:
            raise ToolException(
                f"Remote executor request failed for tool '{tool_name}': {exc}"
            ) from exc

        if not response_data.get("success", False):
            error_msg = response_data.get("error", "Remote executor reported an error.")
            raise ToolException(f"Tool '{tool_name}' failed remotely: {error_msg}")

        stream_timeout = float(os.getenv("REMOTE_TOOL_STREAM_TIMEOUT", "15"))
        consumed_messages = await self._consume_stream_messages(
            topic=stream_topic,
            timeout=stream_timeout,
            max_messages=20,
            message_id=message_id,
        )

        if consumed_messages:
            final_payload = consumed_messages[-1]
            return self._deserialize_remote_result(
                final_payload.get("result"),
                final_payload.get("result_type", "Unknown"),
            )

        if response_data.get("result") is not None:
            return self._deserialize_remote_result(
                response_data.get("result"),
                response_data.get("result_type", "Unknown"),
            )

        raise ToolException(
            f"Tool '{tool_name}' executed remotely but no result was published."
        )

    def _coerce_tool_inputs(
        self,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """Map LangChain tool args into component input values."""
        if kwargs:
            return kwargs

        if not args:
            return {}

        first = args[0]
        if isinstance(first, dict):
            return first

        input_keys: list[str] = metadata.get("_tool_input_keys") or []
        if input_keys:
            return {input_keys[0]: first}
        return {"input_value": first}

    @staticmethod
    async def _consume_stream_messages(
        topic: str | None,
        timeout: float = 5.0,
        max_messages: int = 10,
        message_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Consume NATS messages for remote tool executions."""
        if not topic:
            logger.warning("[NATS] Missing stream topic, cannot consume messages.")
            return []

        try:
            import nats
            from nats.js import JetStreamContext
        except ImportError:
            logger.warning("[NATS] nats-py is not installed; skipping stream consumption.")
            return []

        nc = None
        messages: list[dict[str, Any]] = []
        try:
            nats_url = os.getenv("NATS_URL", "nats://localhost:4222")
            nc = await nats.connect(nats_url)
            js: JetStreamContext = nc.jetstream()

            queue: asyncio.Queue = asyncio.Queue()

            async def _message_handler(msg):
                try:
                    payload = json.loads(msg.data.decode("utf-8"))
                except Exception:  # noqa: BLE001
                    payload = {"raw": msg.data}
                await queue.put(payload)

            sub = await js.subscribe(subject=topic, cb=_message_handler, durable=False)

            try:
                while len(messages) < max_messages:
                    payload = await asyncio.wait_for(queue.get(), timeout=timeout)
                    if message_id and payload.get("message_id") != message_id:
                        continue
                    messages.append(payload)
                    if message_id:
                        break
            except asyncio.TimeoutError:
                logger.debug("[NATS] Timeout waiting for messages on %s", topic)
            finally:
                await sub.unsubscribe()

        except Exception as exc:  # noqa: BLE001
            logger.warning("[NATS] Failed to consume stream %s: %s", topic, exc)
        finally:
            if nc and not nc.is_closed:
                await nc.close()

        return messages

    @staticmethod
    def _deserialize_remote_result(result: Any, result_type: str) -> Any:
        """Reconstruct Langflow primitives from serialized executor payloads."""
        if not isinstance(result, dict):
            return result

        try:
            from lfx.schema.message import Message
            from lfx.schema.data import Data

            if result_type == "Message" or any(
                key in result for key in ["text", "sender", "category", "timestamp", "session_id"]
            ):
                if "timestamp" in result and isinstance(result["timestamp"], str):
                    timestamp = result["timestamp"].replace("T", " ")
                    timestamp = timestamp.replace("+00:00", " UTC").replace("Z", " UTC")
                    result["timestamp"] = timestamp

                data_value = result.get("data")
                if isinstance(data_value, dict):
                    return Message(data=data_value, **{k: v for k, v in result.items() if k != "data"})
                return Message(**result)

            if result_type == "Data" or "data" in result or "text_key" in result:
                return Data(**result)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Failed to deserialize remote result (%s): %s", result_type, exc)

        if isinstance(result, list):
            return [AgentComponent._deserialize_remote_result(item, "Unknown") for item in result]

        return {k: AgentComponent._deserialize_remote_result(v, "Unknown") for k, v in result.items()}
