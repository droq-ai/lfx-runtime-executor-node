from __future__ import annotations

from typing import Any

from lfx.components._importing import import_mod

_dynamic_imports = {
    "CodeComponent": "code",
    "WaitComponent": "wait",
}

__all__ = ["CodeComponent", "WaitComponent"]


def __getattr__(attr_name: str) -> Any:
    if attr_name not in _dynamic_imports:
        raise AttributeError(f"module '{__name__}' has no attribute '{attr_name}'")
    try:
        result = import_mod(attr_name, _dynamic_imports[attr_name], __spec__.parent)
    except (ModuleNotFoundError, ImportError, AttributeError) as e:
        raise AttributeError(f"Could not import '{attr_name}' from '{__name__}': {e}") from e
    globals()[attr_name] = result
    return result


def __dir__() -> list[str]:
    return list(__all__)

