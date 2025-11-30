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
                "Optional Data/DataFrame inputs accessible inside your code via the variables "
                "`data_inputs`, `inputs`, `first_input`, and `input_data`."
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
            
            # Inject data inputs into globals with multiple variable names for convenience
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

            # Only include serializable locals (skip complex objects)
            if python_repl.locals:
                serializable_locals = {}
                for k, v in python_repl.locals.items():
                    if k.startswith("__"):
                        continue
                    # Only include JSON-serializable types
                    if isinstance(v, (str, int, float, bool, type(None))):
                        serializable_locals[k] = v
                    elif isinstance(v, (list, tuple)):
                        try:
                            # Check if list is serializable
                            import json
                            json.dumps(v)
                            serializable_locals[k] = v
                        except (TypeError, ValueError):
                            serializable_locals[k] = str(v)[:200]
                    elif isinstance(v, dict):
                        try:
                            import json
                            json.dumps(v)
                            serializable_locals[k] = v
                        except (TypeError, ValueError):
                            serializable_locals[k] = f"<dict with {len(v)} keys>"
                    else:
                        # Convert non-serializable to string representation
                        serializable_locals[k] = f"<{type(v).__name__}>"
                if serializable_locals:
                    payload["locals"] = serializable_locals
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
        """Prepare and normalize the data_inputs for injection into Python globals."""
        data_inputs = getattr(self, "data_inputs", None) or []
        
        # Ensure it's a list
        if not isinstance(data_inputs, list):
            data_inputs = [data_inputs]
        
        prepared = []
        for item in data_inputs:
            if item is None:
                continue
            if isinstance(item, Data):
                # Keep Data objects as-is, but also expose their .data dict
                prepared.append(item.data if hasattr(item, 'data') else item)
            elif isinstance(item, DataFrame):
                # Convert DataFrame to dict for easy access
                prepared.append(item.to_dict() if hasattr(item, 'to_dict') else item)
            elif isinstance(item, Message):
                # Convert Message to dict
                prepared.append(item.model_dump() if hasattr(item, 'model_dump') else item)
            elif isinstance(item, dict):
                prepared.append(item)
            else:
                prepared.append(item)
        
        return prepared
