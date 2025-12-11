"""DFX Split Component - Splits a Data/Message containing a list into multiple Data outputs.

This component takes a Data or Message object containing a list (e.g., in data.items or text as JSON)
and splits it into multiple Data outputs, one for each item in the list.
"""

import copy
import inspect
import json
from typing import Any

from lfx.custom.custom_component.component import Component
from lfx.io import DataInput, IntInput, Output
from lfx.schema.data import Data
from lfx.schema.dataframe import DataFrame
from lfx.schema.message import Message


class DFXSplitComponent(Component):
    """Split a Data/Message containing a list into multiple Data outputs.
    
    This component extracts a list from the input (Data or Message) and creates
    one Data output for each item in the list. The outputs are dynamically
    created based on the list length.
    """
    
    display_name = "DFX Split"
    description = "Split a Data/Message containing a list into multiple Data outputs. Outputs are dynamically created based on the list length."
    icon = "Split"
    name = "DFXSplit"
    documentation: str = "https://docs.langflow.org/components-data#split"
    
    inputs = [
        DataInput(
            name="input_data",
            display_name="Input Data",
            info=(
                "Data or Message object containing a list to split. "
                "The list can be in data.items, data field, or text field as JSON."
            ),
            required=True,
            input_types=["Data", "Message"],
        ),
        IntInput(
            name="num_outputs",
            display_name="Number of Outputs",
            info=(
                "Number of outputs to create. This determines how many output ports "
                "will be created. If the input list has more items than this number, "
                "only the first N items will be used. If the list has fewer items, "
                "the outputs will still be created (some may be empty)."
            ),
            value=1,
            required=False,
            real_time_refresh=True,
        ),
    ]
    
    outputs = [
        Output(
            display_name="Output",
            name="output",
            method="split",
            types=["Data"],
            group_outputs=True,  # Make outputs appear as distinct handles
        ),
    ]
    
    def __init__(self, **kwargs):
        """Initialize the component and extract list from input."""
        super().__init__(**kwargs)
        self._items_list: list[Any] = []
        # Don't extract here - wait for input to be set
    
    def _extract_list_from_input(self):
        """Extract the list from the input data."""
        input_data = getattr(self, "input_data", None)
        if input_data is None:
            return
        
        # Handle Message
        if isinstance(input_data, Message):
            # Try to get list from data field
            if input_data.data and isinstance(input_data.data, dict):
                # Check for items key
                if "items" in input_data.data and isinstance(input_data.data["items"], list):
                    self._items_list = input_data.data["items"]
                    return
                # Check if data itself is a list structure
                if isinstance(input_data.data, list):
                    self._items_list = input_data.data
                    return
            # Try to parse text as JSON
            if input_data.text:
                try:
                    parsed = json.loads(input_data.text)
                    if isinstance(parsed, list):
                        self._items_list = parsed
                        return
                    elif isinstance(parsed, dict) and "items" in parsed and isinstance(parsed["items"], list):
                        self._items_list = parsed["items"]
                        return
                except (json.JSONDecodeError, ValueError):
                    pass
        
        # Handle Data
        elif isinstance(input_data, Data):
            # Check data.items
            if input_data.data and isinstance(input_data.data, dict):
                if "items" in input_data.data and isinstance(input_data.data["items"], list):
                    self._items_list = input_data.data["items"]
                    return
                # Check if data itself contains a list
                for key, value in input_data.data.items():
                    if isinstance(value, list) and len(value) > 0:
                        self._items_list = value
                        return
            # Try to parse text as JSON
            if hasattr(input_data, 'get_text'):
                text_value = input_data.get_text() or ""
                if text_value:
                    try:
                        parsed = json.loads(text_value)
                        if isinstance(parsed, list):
                            self._items_list = parsed
                            return
                        elif isinstance(parsed, dict) and "items" in parsed and isinstance(parsed["items"], list):
                            self._items_list = parsed["items"]
                            return
                    except (json.JSONDecodeError, ValueError):
                        pass
        
        # Handle dict directly
        elif isinstance(input_data, dict):
            if "items" in input_data and isinstance(input_data["items"], list):
                self._items_list = input_data["items"]
                return
            # Check if it's a list structure
            for key, value in input_data.items():
                if isinstance(value, list) and len(value) > 0:
                    self._items_list = value
                    return
    
    def update_outputs(self, frontend_node: dict, field_name: str, field_value: Any) -> dict:
        """Dynamically create outputs based on the number of outputs specified by the user."""
        # Update outputs when input_data or num_outputs changes
        if field_name in ("input_data", "num_outputs") or field_name is None:
            # Try to extract list from input_data
            # field_value might be the actual data or None if it's a connected edge
            if field_name == "input_data":
                input_data_to_check = field_value if field_value is not None else getattr(self, "input_data", None)
                
                if input_data_to_check is not None:
                    # Handle list input (when multiple inputs are connected)
                    if isinstance(input_data_to_check, list) and len(input_data_to_check) > 0:
                        temp_input = input_data_to_check[0]
                    else:
                        temp_input = input_data_to_check
                    
                    # Temporarily set to extract list
                    original_input = getattr(self, "input_data", None)
                    try:
                        self.input_data = temp_input
                        self._extract_list_from_input()
                    finally:
                        self.input_data = original_input
                else:
                    # If no input data yet, try to extract from current state
                    self._extract_list_from_input()
            elif field_name == "num_outputs":
                # When num_outputs changes, try to extract list from current input_data if available
                self._extract_list_from_input()
            elif field_name is None:
                # Called on component load - try to extract from current state
                self._extract_list_from_input()
            
            # Get the number of outputs requested by the user
            # If field_name is "num_outputs", use field_value, otherwise get from component
            if field_name == "num_outputs" and field_value is not None:
                num_outputs = int(field_value) if isinstance(field_value, (int, str)) else getattr(self, "num_outputs", 1)
            else:
                num_outputs = getattr(self, "num_outputs", None)
            
            if num_outputs is None or num_outputs <= 0:
                num_outputs = 1  # Default to 1 output
            
            # Always use num_outputs as the number of outputs to create
            # The user decides how many outputs they want, regardless of available items
            # If there are more items than outputs, only the first N items will be used
            # If there are fewer items than outputs, some outputs may not have data (but will still be created)
            actual_num_outputs = num_outputs
            
            # Clear existing outputs
            frontend_node["outputs"] = []
            
            # Create outputs based on the number specified by the user
            if actual_num_outputs > 0:
                for i in range(actual_num_outputs):
                    frontend_node["outputs"].append(
                        Output(
                            display_name=f"Output {i + 1}",
                            name=f"output_{i + 1}",
                            method="get_item",
                            types=["Data"],
                            group_outputs=True,  # Make outputs appear as distinct handles
                        ).to_dict()
                    )
            else:
                # Default: create a single output if we can't determine the list yet
                frontend_node["outputs"].append(
                    Output(
                        display_name="Output",
                        name="output",
                        method="split",
                        types=["Data"],
                    ).to_dict()
                )
        
        return frontend_node
    
    def build(self):
        """Build the component and try to update outputs if input data is available."""
        # Try to extract list from input_data if available
        self._extract_list_from_input()
        return self.split
    
    def split(self) -> Data:
        """Split the input into multiple Data objects.
        
        Returns the first item as a Data object. For multiple outputs, use get_item().
        """
        # Extract list from input
        self._extract_list_from_input()
        
        if not self._items_list:
            self.log("No list found in input data")
            return Data(data={"error": "No list found in input data"})
        
        self.log(f"Found {len(self._items_list)} items in input")
        
        # Return first item as default
        item = self._items_list[0]
        if isinstance(item, dict):
            return Data(data=copy.deepcopy(item))
        elif isinstance(item, (str, int, float, bool)):
            return Data(data={"value": item})
        else:
            return Data(data={"item": item, "index": 0})
    
    def get_item(self) -> Data:
        """Get a specific item from the split list.
        
        This method is called for each dynamic output. It determines which item
        to return based on the current output name (output_1, output_2, etc.).
        Only returns items up to the num_outputs limit.
        """
        # Extract list from input
        self._extract_list_from_input()
        
        if not self._items_list:
            self.log("No list found in input data")
            return Data(data={"error": "No list found in input data"})
        
        # Get the number of outputs requested
        num_outputs = getattr(self, "num_outputs", None)
        if num_outputs is None or num_outputs <= 0:
            num_outputs = len(self._items_list)  # Use all items if not specified
        
        # Limit items to the first N items based on num_outputs
        limited_items = self._items_list[:num_outputs]
        
        # Get the current output name to determine which item to return
        # _current_output should be set by the framework before calling this method
        # It may also be in config if called via executor node
        current_output = getattr(self, "_current_output", "")
        
        # If not set, try to get from config (for executor node execution)
        if not current_output:
            config = getattr(self, "__config", {}) or {}
            current_output = config.get("_current_output", "")
        
        # Debug logging
        self.log(f"get_item called, _current_output='{current_output}'")
        
        # If _current_output is still not set, this is a problem
        # Try to get it from the outputs_map by checking which outputs use this method
        if not current_output:
            self.log("Warning: _current_output is not set, trying to determine from context")
            # Check if we can get it from the execution context
            # This is a fallback and may not work correctly
            outputs_map = getattr(self, "_outputs_map", {})
            # Get all outputs that use get_item method
            matching_outputs = [
                name for name, output_obj in outputs_map.items()
                if output_obj.method == "get_item" and name.startswith("output_")
            ]
            if matching_outputs:
                # Sort to get consistent ordering
                matching_outputs.sort()
                # Use the first one as fallback (not ideal, but better than error)
                current_output = matching_outputs[0]
                self.log(f"Using fallback output: {current_output} (this may be incorrect!)")
            else:
                self.log("Error: Could not determine output name, using first item")
                current_output = "output_1"  # Default fallback
        
        # Extract index from output name (e.g., "output_1" -> 0, "output_2" -> 1, or "item_1" -> 0)
        index = 0  # Default to first item
        try:
            if current_output:
                if current_output.startswith("output_"):
                    index_str = current_output.replace("output_", "")
                    index = int(index_str) - 1  # Convert to 0-based index
                elif current_output.startswith("item_"):
                    index_str = current_output.replace("item_", "")
                    index = int(index_str) - 1  # Convert to 0-based index
                self.log(f"Current output: {current_output}, extracted index: {index}")
            else:
                self.log("Warning: Could not determine current output name, using first item")
        except (ValueError, AttributeError) as e:
            # If we can't parse, default to 0
            self.log(f"Error parsing output name '{current_output}': {e}, using first item")
            pass
        
        # Ensure index is valid within the limited items
        if index < 0 or index >= len(limited_items):
            self.log(f"Invalid index {index} for limited list of length {len(limited_items)}")
            if limited_items:
                index = 0
            else:
                return Data(data={"error": "No items available"})
        
        # Get the item at the specified index from the limited list
        item = limited_items[index]
        self.log(f"Returning item {index + 1} (index {index}) of {len(limited_items)} (from {len(self._items_list)} total items)")
        
        # Convert item to Data object
        if isinstance(item, dict):
            return Data(data=copy.deepcopy(item))
        elif isinstance(item, (str, int, float, bool)):
            return Data(data={"value": item})
        else:
            return Data(data={"item": item, "index": index})
