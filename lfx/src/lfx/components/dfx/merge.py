"""DFX Merge Component - Merges multiple Data inputs into a single Data output.

This component waits for all inputs (is_list=True) and combines them into a single
Data object, preserving all data without loss.
"""

from typing import Any

from lfx.custom.custom_component.component import Component
from lfx.io import DataInput, Output
from lfx.schema.data import Data
from lfx.schema.dataframe import DataFrame
from lfx.schema.message import Message


class DFXMergeComponent(Component):
    """Merge multiple Data inputs into a single Data output.
    
    This component collects all data inputs (which can be Data, DataFrame, or Message objects),
    preserves all data from each input, and combines them into a single Data object.
    All data is preserved - nothing is lost during the merge.
    
    The component waits for all inputs before executing (is_list=True ensures this).
    """
    
    display_name = "DFX Merge"
    description = "Merge multiple Data/DataFrame/Message inputs into a single Data output. Waits for all inputs before merging. Preserves all data without loss."
    icon = "Merge"
    name = "DFXMerge"
    documentation: str = "https://docs.langflow.org/components-data#merge"
    
    inputs = [
        DataInput(
            name="data_inputs",
            display_name="Data Inputs",
            info=(
                "Multiple Data/DataFrame/Message inputs to merge. "
                "The component waits for all inputs before merging them into a single Data object. "
                "All data is preserved without loss."
            ),
            is_list=True,
            required=True,
            input_types=["Data", "DataFrame", "Message"],
        ),
    ]
    
    outputs = [
        Output(
            display_name="Merged Data",
            name="result",
            types=["Data"],
            method="merge",
        ),
    ]
    
    def build(self):
        """Return the merge method for execution."""
        return self.merge
    
    def merge(self) -> Data:
        """Merge multiple data inputs into a single Data output, preserving all data.
        
        This method:
        1. Collects all data_inputs (which should be a list due to is_list=True)
        2. Preserves all data from each input (handles Data, DataFrame, Message)
        3. Extracts and preserves nested structures (e.g., Data objects containing DataFrames)
        4. Combines all items into a single list
        5. Returns a Data object containing all merged data
        
        Returns:
            Data: Contains all merged data as a list in the data field.
                The data field will contain a list of all items from all inputs.
        """
        # Get data_inputs - should be a list due to is_list=True
        data_inputs = getattr(self, "data_inputs", None) or []
        
        # Ensure it's a list
        if not isinstance(data_inputs, list):
            if data_inputs is None:
                data_inputs = []
            else:
                data_inputs = [data_inputs]
        
        # If it's an Input object, get its value
        if hasattr(data_inputs, 'value'):
            data_inputs = data_inputs.value
            if not isinstance(data_inputs, list):
                data_inputs = [data_inputs] if data_inputs else []
        
        # Also check _attributes as fallback
        if not data_inputs:
            data_inputs = getattr(self, "_attributes", {}).get("data_inputs", None) or []
            if not isinstance(data_inputs, list):
                data_inputs = [data_inputs] if data_inputs else []
        
        self.log(f"Merging {len(data_inputs)} input(s)")
        
        # Collect all data items - preserving everything
        all_items = []
        
        for idx, input_data in enumerate(data_inputs):
            if input_data is None:
                continue
                
            # Handle DataFrame - preserve all rows
            if isinstance(input_data, DataFrame):
                items = input_data.to_dict(orient="records")
                if isinstance(items, list):
                    all_items.extend(items)
                    self.log(f"Input {idx}: DataFrame with {len(items)} row(s)")
                else:
                    all_items.append(items)
                    self.log(f"Input {idx}: DataFrame with 1 row")
                
            # Handle Message - preserve data and text
            elif isinstance(input_data, Message):
                if input_data.data:
                    all_items.append(input_data.data)
                    self.log(f"Input {idx}: Message with data")
                elif input_data.text:
                    all_items.append({"text": input_data.text})
                    self.log(f"Input {idx}: Message with text")
                else:
                    # Preserve empty message as well
                    all_items.append({"message": None})
                    self.log(f"Input {idx}: Empty Message")
                
            # Handle Data object - preserve all fields
            elif isinstance(input_data, Data):
                # Preserve the entire data structure
                if input_data.data is not None:
                    data_content = input_data.data
                    
                    # If it's a dict with "type": "DataFrame" and "data" key, extract the data list
                    if isinstance(data_content, dict):
                        if data_content.get("type") == "DataFrame" and "data" in data_content:
                            df_data = data_content["data"]
                            if isinstance(df_data, list):
                                # Preserve all items from the DataFrame
                                all_items.extend(df_data)
                                self.log(f"Input {idx}: Data with DataFrame containing {len(df_data)} item(s)")
                            else:
                                # Single item DataFrame
                                all_items.append(df_data)
                                self.log(f"Input {idx}: Data with DataFrame containing 1 item")
                        else:
                            # Regular dict data - preserve as-is
                            all_items.append(data_content)
                            self.log(f"Input {idx}: Data with dict data")
                    elif isinstance(data_content, list):
                        # List data - preserve all items
                        all_items.extend(data_content)
                        self.log(f"Input {idx}: Data with list containing {len(data_content)} item(s)")
                    else:
                        # Other data types - preserve as-is
                        all_items.append(data_content)
                        self.log(f"Input {idx}: Data with {type(data_content).__name__} data")
                elif input_data.text:
                    # Preserve text
                    all_items.append({"text": input_data.text})
                    self.log(f"Input {idx}: Data with text")
                else:
                    # Preserve empty Data object
                    all_items.append({})
                    self.log(f"Input {idx}: Empty Data object (preserved as empty dict)")
                    
            # Handle dict directly - preserve as-is
            elif isinstance(input_data, dict):
                # Check if it's a DataFrame structure
                if input_data.get("type") == "DataFrame" and "data" in input_data:
                    df_data = input_data["data"]
                    if isinstance(df_data, list):
                        # Extract all items from DataFrame structure
                        all_items.extend(df_data)
                        self.log(f"Input {idx}: Dict with DataFrame containing {len(df_data)} item(s)")
                    else:
                        # Single item
                        all_items.append(df_data)
                        self.log(f"Input {idx}: Dict with DataFrame containing 1 item")
                else:
                    # Regular dict - preserve as-is
                    all_items.append(input_data)
                    self.log(f"Input {idx}: Dict data")
                    
            # Handle list directly - preserve all items
            elif isinstance(input_data, list):
                all_items.extend(input_data)
                self.log(f"Input {idx}: List with {len(input_data)} item(s)")
                
            # Handle other types - preserve by converting to dict
            else:
                all_items.append({"value": input_data, "type": type(input_data).__name__})
                self.log(f"Input {idx}: Other type ({type(input_data).__name__}) - preserved")
        
        self.log(f"Merged {len(all_items)} total item(s) from {len(data_inputs)} input(s)")
        
        # Return as Data object with all merged data in the data field
        # Data requires data to be a dict, so we wrap the list in a dictionary
        # This preserves all data without wrapping in DataFrame structure
        return Data(data={"items": all_items})
