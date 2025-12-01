from lfx.custom.custom_component.component import Component
from lfx.io import CodeInput, Output
from lfx.schema.data import Data
from lfx.schema.message import Message


class CodeComponent(Component):
    display_name = "Code"
    description = "Simple multi-language code editor for writing or sharing snippets."
    icon = "Code"
    name = "Code"

    inputs = [
        CodeInput(
            name="code",
            display_name="Code",
            info="Write any text or code snippet. No execution is performed.",
            value="print('Hello, world!')",
        ),
    ]

    outputs = [
        Output(
            name="text_output",
            display_name="Code (Text)",
            method="return_text",
        ),
        Output(
            name="data_output",
            display_name="Code (Data)",
            method="return_data",
        ),
    ]

    def return_text(self) -> Message:
        content = getattr(self, "code", "") or ""
        return Message(text=str(content))

    def return_data(self) -> Data:
        content = getattr(self, "code", "") or ""
        return Data(text=str(content))


