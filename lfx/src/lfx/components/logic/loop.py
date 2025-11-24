from copy import deepcopy
from typing import Any

from lfx.custom.custom_component.component import Component
from lfx.inputs.inputs import HandleInput
from lfx.schema.data import Data
from lfx.schema.dataframe import DataFrame
from lfx.schema.message import Message
from lfx.template.field.base import Output


class LoopComponent(Component):
    LOOP_STATE_KEY = "__loop_state"
    display_name = "Loop"
    description = (
        "Iterates over a list of Data objects, outputting one item at a time and aggregating results from loop inputs."
    )
    documentation: str = "https://docs.langflow.org/components-logic#loop"
    icon = "infinity"

    inputs = [
        HandleInput(
            name="data",
            display_name="Inputs",
            info="The initial list of Data objects or DataFrame to iterate over.",
            input_types=["DataFrame", "Data"],
        ),
        HandleInput(
            name="next",
            display_name="Next",
            info="Connect the output of the last component in the loop to trigger the next iteration. Any output received will advance to the next item.",
            input_types=["Message", "Data"],
            required=False,
        ),
    ]

    outputs = [
        Output(display_name="Item", name="item", method="item_output", allows_loop=True, group_outputs=True),
        Output(display_name="Done", name="done", method="done_output", group_outputs=True),
    ]

    # ------------------------------------------------------------------
    # Internal state helpers
    # ------------------------------------------------------------------
    def _get_state(self) -> dict:
        state = getattr(self, "_loop_state_cache", None)
        if state is None:
            attrs = getattr(self, "_attributes", {}) or {}
            state = attrs.get(self.LOOP_STATE_KEY)
            if not isinstance(state, dict):
                state = {}
            self._loop_state_cache = state
        return state

    def _persist_state(self) -> None:
        state = self._get_state()
        if not hasattr(self, "_attributes") or self._attributes is None:
            self._attributes = {}
        self._attributes[self.LOOP_STATE_KEY] = state

    def _get_index(self) -> int:
        return int(self._get_state().get("index", 0))

    def _set_index(self, value: int) -> None:
        self._get_state()["index"] = max(0, int(value))
        self._persist_state()

    def _get_aggregated(self) -> list[dict]:
        agg = self._get_state().get("aggregated", [])
        if not isinstance(agg, list):
            agg = []
        return agg

    def _set_aggregated(self, values: list[dict]) -> None:
        self._get_state()["aggregated"] = values
        self._persist_state()

    def _reset_state(self) -> None:
        # Invalidate cache first to ensure fresh state
        if hasattr(self, "_loop_state_cache"):
            delattr(self, "_loop_state_cache")
        # Get fresh state (will create new dict if needed)
        state = self._get_state()
        state["index"] = 0
        state["aggregated"] = []
        state["stopped"] = False  # Reset stopped flag
        # Update cache and persist
        self._loop_state_cache = state
        self._persist_state()
        if hasattr(self, "_data_cache"):
            delattr(self, "_data_cache")
    
    def _is_stopped(self) -> bool:
        """Check if the loop has been stopped."""
        return bool(self._get_state().get("stopped", False))
    
    def _set_stopped(self, value: bool = True) -> None:
        """Mark the loop as stopped."""
        self._get_state()["stopped"] = value
        self._persist_state()

    # ------------------------------------------------------------------
    # Data helpers
    # ------------------------------------------------------------------
    def _get_data_list(self) -> list[Data]:
        cache = getattr(self, "_data_cache", None)
        if cache is None:
            cache = self._validate_data(self.data)
            self._data_cache = cache
        return cache

    def _validate_data(self, data):
        if isinstance(data, str):
            parsed = self._try_parse_string(data)
            if parsed is not None:
                data = parsed
        if isinstance(data, DataFrame):
            return data.to_data_list()
        if isinstance(data, Data):
            return [data]
        if isinstance(data, list):
            return [self._coerce_to_data(item) for item in data]
        if isinstance(data, dict):
            return [self._coerce_to_data(data)]
        msg = "The 'data' input must be a DataFrame, a list of Data objects, or a single Data object."
        raise TypeError(msg)

    @staticmethod
    def _try_parse_string(value: str):
        import json

        stripped = value.strip()
        if not stripped:
            return None
        try:
            return json.loads(stripped)
        except (json.JSONDecodeError, ValueError):
            return None

    @staticmethod
    def _coerce_to_data(item):
        if isinstance(item, Data):
            text_key = getattr(item, "text_key", "text") or "text"
            payload = item.data if isinstance(item.data, dict) else {}
            payload = LoopComponent._ensure_payload_text(payload, text_key, getattr(item, "text", None))
            text_value = payload.get(text_key)
            return Data(
                data=payload,
                text_key=text_key,
                default_value=getattr(item, "default_value", ""),
                text=text_value,
            )
        if isinstance(item, dict):
            if "data" in item:
                payload = item.get("data", {})
                text_key = item.get("text_key") or "text"
                default_value = item.get("default_value", "")
                payload = LoopComponent._ensure_payload_text(payload, text_key)
                text = payload.get(text_key) if isinstance(payload, dict) else None
                return Data(data=payload, text_key=text_key, default_value=default_value, text=text)
            payload = LoopComponent._ensure_payload_text(item, "text")
            return Data(data=payload, text_key="text", text=payload.get("text"))
        if isinstance(item, str):
            parsed = LoopComponent._try_parse_string(item)
            if parsed is not None:
                return LoopComponent._coerce_to_data(parsed)
        return Data(data={"value": item}, text_key="value")

    @staticmethod
    def _serialize_data(item: Data | dict) -> dict:
        if isinstance(item, Data):
            payload = deepcopy(item.data)
            text_key = getattr(item, "text_key", "text") or "text"
            text_value = getattr(item, "text", None)
            payload = LoopComponent._ensure_payload_text(payload, text_key, text_value)
            return {
                "data": payload,
                "text_key": text_key,
                "default_value": getattr(item, "default_value", ""),
            }
        if isinstance(item, dict):
            return {"data": deepcopy(item)}
        return {"data": item}

    @staticmethod
    def _deserialize_data(payload) -> Data:
        if isinstance(payload, Data):
            return payload
        if isinstance(payload, dict):
            data = payload.get("data", {})
            text_key = payload.get("text_key", "text")
            default_value = payload.get("default_value", "")
            data = LoopComponent._ensure_payload_text(data, text_key)
            return Data(data=data, text_key=text_key, default_value=default_value)
        return Data(data={"value": payload})

    @staticmethod
    def _ensure_payload_text(payload, text_key: str, explicit_text: str | None = None):
        if not isinstance(payload, dict):
            return payload
        if explicit_text:
            payload[text_key] = explicit_text
            if text_key != "text":
                payload.setdefault("text", explicit_text)
        else:
            value = payload.get(text_key)
            if not value:
                import json

                payload[text_key] = json.dumps(payload, ensure_ascii=False)
                if text_key != "text":
                    payload.setdefault("text", payload[text_key])
        return payload

    # ------------------------------------------------------------------
    # Core component logic
    # ------------------------------------------------------------------
    def initialize_data(self) -> None:
        state = self._get_state()
        raw = getattr(self, "data", None)
        data_list = self._get_data_list()
        preview = data_list[0].data if data_list else None
        current_index = self._get_index()        
        # Reset state ONLY if:
        #   1. Index is beyond bounds (corrupted state)
        #   2. Do NOT reset if stopped=True (loop ended) - backend will handle stopping
        #   3. Do NOT reset if index == length (reached end) - that's valid state, just return empty Data
        #   4. The reset when stopped=True should only happen on the BACKEND side when starting a new run
        if current_index > len(data_list):
            self._reset_state()
            current_index = 0

    def evaluate_stop_loop(self) -> bool:
        data_length = len(self._get_data_list())
        return self._get_index() >= data_length

    def item_output(self) -> Data:
        """
        Executor version: Stateless execution - just return item at current index and increment.
        Backend orchestrates when to call this method.
        """
        self.initialize_data()
        
        # If already stopped, return immediately without processing
        if self._is_stopped():
            return Data(text="")
        
        data_list = self._get_data_list()
        current_index = self._get_index()
        
        # Log state for debugging
        state = self._get_state()
        # Executor is stateless - just execute what backend asks
        # Backend should check bounds before calling executor
        if current_index >= len(data_list):
            # Mark as stopped so backend knows
            self._set_stopped(True)
            return Data(text="")

        try:
            current_item = data_list[current_index]
        except IndexError:
            self._set_stopped(True)
            return Data(text="")
        
        current_item = self._coerce_to_data(current_item)

        # Increment index and add to aggregated for next iteration
        serialized = self._get_aggregated()
        serialized.append(self._serialize_data(current_item))
        self._set_aggregated(serialized)
        new_index = current_index + 1
        self._set_index(new_index)
        return current_item

    def done_output(self) -> DataFrame:
        self.initialize_data()
        serialized = self._get_aggregated()
        if not serialized:
            rows = [self._serialize_data(item) for item in self._get_data_list()]
        else:
            rows = serialized
        self._reset_state()
        if not rows:
            return DataFrame([])
        data_objects = [self._deserialize_data(entry) for entry in rows]
        return DataFrame(data_objects)

    def loop_variables(self):
        return self._get_data_list(), self._get_index()

    def aggregated_output(self, current_item: Data | None = None):
        if current_item is not None:
            serialized = self._get_aggregated()
            serialized.append(self._serialize_data(current_item))
            self._set_aggregated(serialized)
        return [self._deserialize_data(entry) for entry in self._get_aggregated()]

    def build_config(self):
        """Build config and mark 'next' input as a loop input that accepts cycles."""
        config = super().build_config()
        # Mark the 'next' input as a loop input by adding output_types
        # This tells the frontend it can accept cycle connections
        if "next" in config and isinstance(config["next"], dict):
            config["next"]["output_types"] = ["Message", "Data", "DataFrame", "str", "int", "float", "bool"]
        return config

    def update_build_config(self, build_config: dict, field_value: Any, field_name: str | None = None) -> dict:
        """Update build config to mark 'next' input as a loop input that accepts cycles."""
        build_config = super().update_build_config(build_config, field_value, field_name)
        # Mark the 'next' input as a loop input by adding output_types
        # This tells the frontend it can accept cycle connections
        if "next" in build_config and isinstance(build_config["next"], dict):
            build_config["next"]["output_types"] = ["Message", "Data", "DataFrame", "str", "int", "float", "bool"]
        return build_config

    def update_dependency(self):
        graph = getattr(self, "graph", None)
        vertex = getattr(self, "_vertex", None)
        if graph is None or vertex is None:
            return
        try:
            item_dependency_id = self.get_incoming_edge_by_target_param("item")
        except ValueError:
            return
        if not item_dependency_id:
            return
        run_manager = getattr(graph, "run_manager", None)
        if run_manager is None:
            return
        predecessors = run_manager.run_predecessors.setdefault(vertex.id, [])
        if item_dependency_id not in predecessors:
            predecessors.append(item_dependency_id)
            dependents = run_manager.run_map.setdefault(item_dependency_id, [])
            if vertex.id not in dependents:
                dependents.append(vertex.id)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _should_emit_next(self) -> bool:
        """
        Executor version: Stateless - just check if 'next' input has a value.
        Backend handles stop conditions and orchestration.
        """
        value = getattr(self, "next", None)
        if value is None:
            # Not connected - auto-advance (backend will check stop conditions)
            return True
        # Any value received (Message, Data, str, int, float, bool, etc.) triggers next iteration
        # Check for common Langflow types
        if isinstance(value, (Message, Data)):
            return True
        # Check for dict (serialized Message/Data)
        if isinstance(value, dict):
            return True
        # Check for primitive types (str, int, float, bool)
        if isinstance(value, (str, int, float, bool)):
            # Empty string or zero values still trigger (presence matters, not value)
            return True
        # Fallback: any truthy value triggers next
        return bool(value)
