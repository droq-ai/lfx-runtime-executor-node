import json
from typing import Any

from lfx.custom import Component
from lfx.io import BoolInput, HandleInput, Output, TabInput
from lfx.schema import Data, DataFrame, Message

MIN_CSV_LINES = 2


def convert_to_message(v) -> Message:
    """Convert input to Message type.

    Args:
        v: Input to convert (Message, Data, DataFrame, dict, or str)

    Returns:
        Message: Converted Message object
    """
    if isinstance(v, Message):
        print(f"[TypeConvert] Input already Message: {v}", flush=True)
        return v
    if isinstance(v, Data):
        payload = v.data.copy()
        text_value = v.get_text() or ""
        if ((not text_value) or text_value == v.default_value) and payload:
            import json

            text_value = json.dumps(payload, ensure_ascii=False)
        print(f"[TypeConvert] Converting Data to Message: {payload} -> {text_value}", flush=True)
        return Message(text=text_value, data=payload)
    if isinstance(v, DataFrame):
        print(f"[TypeConvert] Converting DataFrame to Message: {v.to_dict()}", flush=True)
        return v.to_message()
    if isinstance(v, dict):
        print(f"[TypeConvert] Converting dict to Message: {v}", flush=True)
        return Data(data=v).to_message()
    if isinstance(v, str):
        print(f"[TypeConvert] Converting str to Message: {v}", flush=True)
        return Message(text=v)
    if hasattr(v, "to_message"):
        print(f"[TypeConvert] Using custom to_message for {v}", flush=True)
        return v.to_message()
    # Fallback: wrap arbitrary objects as text
    print(f"[TypeConvert] Fallback Message conversion for {v}", flush=True)
    return Message(text=str(v))


def convert_to_data(v: DataFrame | Data | Message | dict, *, auto_parse: bool) -> Data:
    """Convert input to Data type.

    Args:
        v: Input to convert (Message, Data, DataFrame, or dict)
        auto_parse: Enable automatic parsing of structured data (JSON/CSV)

    Returns:
        Data: Converted Data object
    """
    if isinstance(v, dict):
        print(f"[TypeConvert] Converting dict to Data: {v}", flush=True)
        return Data(v)
    if isinstance(v, Message):
        payload = v.data.copy() if isinstance(v.data, dict) else {}
        if "text" not in payload:
            text_content = ""
            if isinstance(v.text, str):
                text_content = v.text
            payload["text"] = text_content
        payload["metadata"] = payload.get("metadata", {})
        payload["metadata"].update(
            {
                "sender": v.sender,
                "sender_name": v.sender_name,
                "session_id": v.session_id,
                "context_id": v.context_id,
                "timestamp": v.timestamp,
                "flow_id": v.flow_id,
            }
        )
        data = Data(data=payload, text_key="text")
        print(f"[TypeConvert] Converting Message to Data: {payload}", flush=True)
        return parse_structured_data(data) if auto_parse else data

    if isinstance(v, Data):
        print(f"[TypeConvert] Input already Data: {v.data}", flush=True)
        return v
    print(f"[TypeConvert] Using custom to_data for {v}", flush=True)
    return v.to_data()


def convert_to_dataframe(v: DataFrame | Data | Message | dict, *, auto_parse: bool) -> DataFrame:
    """Convert input to DataFrame type.

    Args:
        v: Input to convert (Message, Data, DataFrame, or dict)
        auto_parse: Enable automatic parsing of structured data (JSON/CSV)

    Returns:
        DataFrame: Converted DataFrame object
    """
    import pandas as pd

    if isinstance(v, dict):
        print(f"[TypeConvert] Converting dict to DataFrame: {v}", flush=True)
        return DataFrame([v])
    if isinstance(v, DataFrame):
        print(f"[TypeConvert] Input already DataFrame", flush=True)
        return v
    # Handle pandas DataFrame
    if isinstance(v, pd.DataFrame):
        # Convert pandas DataFrame to our DataFrame by creating Data objects
        return DataFrame(data=v)

    if isinstance(v, Message):
        data = Data(data={"text": v.data["text"]})
        print(f"[TypeConvert] Converting Message to DataFrame: {data.data}", flush=True)
        return parse_structured_data(data).to_dataframe() if auto_parse else data.to_dataframe()
    # For other types, call to_dataframe method
    print(f"[TypeConvert] Using custom to_dataframe for {v}", flush=True)
    return v.to_dataframe()


def parse_structured_data(data: Data) -> Data:
    """Parse structured data (JSON, CSV) from Data's text field.

    Args:
        data: Data object with text content to parse

    Returns:
        Data: Modified Data object with parsed content or original if parsing fails
    """
    raw_text = data.get_text() or ""
    text = raw_text.lstrip("\ufeff").strip()

    # Try JSON parsing first
    parsed_json = _try_parse_json(text)
    if parsed_json is not None:
        return parsed_json

    # Try CSV parsing
    if _looks_like_csv(text):
        try:
            return _parse_csv_to_data(text)
        except Exception:  # noqa: BLE001
            # Heuristic misfire or malformed CSV — keep original data
            return data

    # Return original data if no parsing succeeded
    return data


def _try_parse_json(text: str) -> Data | None:
    """Try to parse text as JSON and return Data object."""
    try:
        parsed = json.loads(text)

        if isinstance(parsed, dict):
            # Single JSON object
            return Data(data=parsed)
        if isinstance(parsed, list) and all(isinstance(item, dict) for item in parsed):
            # Array of JSON objects - create Data with the list
            return Data(data={"records": parsed})

    except (json.JSONDecodeError, ValueError):
        pass

    return None


def _looks_like_csv(text: str) -> bool:
    """Simple heuristic to detect CSV content."""
    lines = text.strip().split("\n")
    if len(lines) < MIN_CSV_LINES:
        return False

    header_line = lines[0]
    return "," in header_line and len(lines) > 1


def _parse_csv_to_data(text: str) -> Data:
    """Parse CSV text and return Data object."""
    from io import StringIO

    import pandas as pd

    # Parse CSV to DataFrame, then convert to list of dicts
    parsed_df = pd.read_csv(StringIO(text))
    records = parsed_df.to_dict(orient="records")

    return Data(data={"records": records})


class TypeConverterComponent(Component):
    display_name = "Type Convert"
    description = "Convert between different types (Message, Data, DataFrame)"
    documentation: str = "https://docs.langflow.org/components-processing#type-convert"
    icon = "repeat"

    inputs = [
        HandleInput(
            name="input_data",
            display_name="Input",
            input_types=["Message", "Data", "DataFrame"],
            info="Accept Message, Data or DataFrame as input",
            required=True,
        ),
        BoolInput(
            name="auto_parse",
            display_name="Auto Parse",
            info="Detect and convert JSON/CSV strings automatically.",
            advanced=True,
            value=False,
            required=False,
        ),
        TabInput(
            name="output_type",
            display_name="Output Type",
            options=["Message", "Data", "DataFrame"],
            info="Select the desired output data type",
            real_time_refresh=True,
            value="Message",
        ),
    ]

    outputs = [
        Output(
            display_name="Message Output",
            name="message_output",
            method="convert_to_message",
        )
    ]

    def update_outputs(self, frontend_node: dict, field_name: str, field_value: Any) -> dict:
        """Dynamically show only the relevant output based on the selected output type."""
        if field_name == "output_type":
            # Start with empty outputs
            frontend_node["outputs"] = []

            # Add only the selected output type
            if field_value == "Message":
                frontend_node["outputs"].append(
                    Output(
                        display_name="Message Output",
                        name="message_output",
                        method="convert_to_message",
                    ).to_dict()
                )
            elif field_value == "Data":
                frontend_node["outputs"].append(
                    Output(
                        display_name="Data Output",
                        name="data_output",
                        method="convert_to_data",
                    ).to_dict()
                )
            elif field_value == "DataFrame":
                frontend_node["outputs"].append(
                    Output(
                        display_name="DataFrame Output",
                        name="dataframe_output",
                        method="convert_to_dataframe",
                    ).to_dict()
                )

        return frontend_node

    def convert_to_message(self) -> Message:
        """Convert input to Message type."""
        input_value = self.input_data[0] if isinstance(self.input_data, list) else self.input_data

        # Handle string input by converting to Message first
        if isinstance(input_value, str):
            input_value = Message(text=input_value)

        result = convert_to_message(input_value)
        self.status = result
        return result

    def convert_to_data(self) -> Data:
        """Convert input to Data type."""
        input_value = self.input_data[0] if isinstance(self.input_data, list) else self.input_data

        # Handle string input by converting to Message first
        if isinstance(input_value, str):
            input_value = Message(text=input_value)

        result = convert_to_data(input_value, auto_parse=self.auto_parse)
        self.status = result
        return result

    def convert_to_dataframe(self) -> DataFrame:
        """Convert input to DataFrame type."""
        input_value = self.input_data[0] if isinstance(self.input_data, list) else self.input_data

        # Handle string input by converting to Message first
        if isinstance(input_value, str):
            input_value = Message(text=input_value)

        result = convert_to_dataframe(input_value, auto_parse=self.auto_parse)
        self.status = result
        return result
