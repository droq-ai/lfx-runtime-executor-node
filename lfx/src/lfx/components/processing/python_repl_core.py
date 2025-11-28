import importlib

from langchain_experimental.utilities import PythonREPL

from lfx.custom.custom_component.component import Component
from lfx.io import DataInput, MultilineInput, Output, StrInput
from lfx.schema.data import Data
from lfx.schema.dataframe import DataFrame
from lfx.schema.message import Message


class PythonREPLComponent(Component):
    display_name = "Python Interpreter"
    description = "Run Python code with optional imports. Use print() to see the output."
    documentation: str = "https://docs.langflow.org/components-processing#python-interpreter"
    icon = "square-terminal"

    inputs = [
        StrInput(
            name="global_imports",
            display_name="Global Imports",
            info="A comma-separated list of modules to import globally, e.g. 'math,numpy,pandas'.",
            value="math,pandas",
            required=True,
        ),
        MultilineInput(
            name="python_code",
            display_name="Python Code",
            info="The Python code to execute. Only modules specified in Global Imports can be used.",
            value="print('Hello, World!')",
            input_types=["Message"],
            tool_mode=True,
            required=True,
        ),
        DataInput(
            name="data_inputs",
            display_name="Data Inputs",
            info=(
                "Optional Data/DataFrame inputs accessible via the variables "
                "`data_inputs`, `inputs`, `first_input`, and `input_data` inside your code."
            ),
            is_list=True,
            required=False,
            input_types=["Data", "DataFrame", "Message"],
        ),
    ]

    outputs = [
        Output(
            display_name="Results",
            name="results",
            type_=Data,
            method="run_python_repl",
        ),
    ]

    def get_globals(self, global_imports: str | list[str]) -> dict:
        """Create a globals dictionary with only the specified allowed imports."""
        global_dict = {}

        try:
            if isinstance(global_imports, str):
                modules = [module.strip() for module in global_imports.split(",")]
            elif isinstance(global_imports, list):
                modules = global_imports
            else:
                msg = "global_imports must be either a string or a list"
                raise TypeError(msg)

            for module in modules:
                try:
                    imported_module = importlib.import_module(module)
                    global_dict[imported_module.__name__] = imported_module
                except ImportError as e:
                    msg = f"Could not import module {module}: {e!s}"
                    raise ImportError(msg) from e

        except Exception as e:
            self.log(f"Error in global imports: {e!s}")
            raise
        else:
            self.log(f"Successfully imported modules: {list(global_dict.keys())}")
            return global_dict

    def run_python_repl(self) -> Data:
        try:
            prepared_inputs = self._prepare_data_inputs()
            globals_ = self.get_globals(self.global_imports)
            globals_["data_inputs"] = prepared_inputs
            globals_["inputs"] = prepared_inputs
            if prepared_inputs:
                globals_["first_input"] = prepared_inputs[0]
                globals_["input_data"] = prepared_inputs

            python_repl = PythonREPL(_globals=globals_)
            result = python_repl.run(self.python_code)
            result = result.strip() if result else ""

            self.log("Code execution completed successfully")
            payload = {"result": result}
            if python_repl.locals:
                payload["locals"] = {k: v for k, v in python_repl.locals.items() if not k.startswith("__")}
            return Data(data=payload)

        except ImportError as e:
            error_message = f"Import Error: {e!s}"
            self.log(error_message)
            return Data(data={"error": error_message})

        except SyntaxError as e:
            error_message = f"Syntax Error: {e!s}"
            self.log(error_message)
            return Data(data={"error": error_message})

        except (NameError, TypeError, ValueError) as e:
            error_message = f"Error during execution: {e!s}"
            self.log(error_message)
            return Data(data={"error": error_message})

    def build(self):
        return self.run_python_repl

    def _prepare_data_inputs(self) -> list[dict | str | Data]:
        data_inputs = getattr(self, "data_inputs", None) or []
        prepared: list[dict | str | Data] = []

        for data in data_inputs:
            if isinstance(data, DataFrame):
                prepared.append(data.to_dict(orient="records"))
            elif isinstance(data, Message):
                prepared.append(data.data if data.data else data.text)
            elif isinstance(data, Data):
                if data.data:
                    prepared.append(data.data)
                elif data.text:
                    prepared.append(data.text)
                else:
                    prepared.append(data)
            else:
                prepared.append(data)
        return prepared
