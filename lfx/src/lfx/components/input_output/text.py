from lfx.base.io.text import TextComponent
from lfx.io import DataInput, MultilineInput, Output
from lfx.schema.message import Message


class TextInputComponent(TextComponent):
    display_name = "Text Input"
    description = "Get user text inputs."
    documentation: str = "https://docs.langflow.org/components-io#text-input"
    icon = "type"
    name = "TextInput"

    inputs = [
        MultilineInput(
            name="input_value",
            display_name="Text",
            info="Text to be passed as input.",
        ),
        DataInput(
            name="trigger",
            display_name="Trigger",
            info=(
                "Optional input used to retrigger this component when connected in loops or other control "
                "components. The incoming value is ignored but ensures a new execution."
            ),
            required=False,
        ),
    ]
    outputs = [
        Output(display_name="Output Text", name="text", method="text_response"),
    ]

    def text_response(self) -> Message:
        return Message(
            text=self.input_value,
        )
