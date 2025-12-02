#!/usr/bin/env python3
"""
Build a JSON mapping of component class names to their module paths.

Scans the lfx/components directory and creates a mapping file.
"""

import ast
import json
import os
from pathlib import Path


def find_component_classes(directory: str) -> dict[str, str]:
    """Scan directory for component classes and build mapping.
    
    Args:
        directory: Root directory to scan (e.g., lfx/src/lfx/components)
        
    Returns:
        Dictionary mapping class names to module paths
    """
    mapping = {}
    components_dir = Path(directory)
    
    if not components_dir.exists():
        print(f"Warning: Components directory not found: {directory}")
        return mapping
    
    # Walk through all Python files in components directory
    for py_file in components_dir.rglob("*.py"):
        # Skip __init__.py and __pycache__
        if py_file.name.startswith("__") or "__pycache__" in str(py_file):
            continue
        
        try:
            # Read and parse the file
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse AST to find component classes
            tree = ast.parse(content, filename=str(py_file))
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    # Check if it's a component class (ends with Component or inherits from Component)
                    class_name = node.name
                    if class_name.endswith("Component") or any(
                        base.id == "Component" or base.id.endswith("Component")
                        for base in node.bases
                        if isinstance(base, ast.Name)
                    ):
                        # Build module path from file path
                        # e.g., lfx/src/lfx/components/input_output/text.py
                        # -> lfx.components.input_output.text
                        rel_path = py_file.relative_to(components_dir.parent.parent)
                        module_parts = list(rel_path.parts[:-1]) + [rel_path.stem]
                        module_path = ".".join(module_parts)
                        
                        # Only add if not already in mapping (first occurrence wins)
                        if class_name not in mapping:
                            mapping[class_name] = module_path
                            print(f"Found: {class_name} -> {module_path}")
        
        except (SyntaxError, UnicodeDecodeError) as e:
            print(f"Warning: Could not parse {py_file}: {e}")
            continue
        except Exception as e:
            print(f"Error processing {py_file}: {e}")
            continue
    
    return mapping


def main():
    """Main function to build component mapping."""
    # Get the script directory
    script_dir = Path(__file__).parent
    
    # Try to find lfx components directory
    lfx_paths = [
        script_dir / "lfx" / "src" / "lfx" / "components",
        script_dir.parent / "app" / "src" / "lfx" / "src" / "lfx" / "components",
    ]
    
    components_dir = None
    for path in lfx_paths:
        if path.exists():
            components_dir = path
            break
    
    if not components_dir:
        print("Error: Could not find lfx/components directory")
        print(f"Tried: {lfx_paths}")
        return 1
    
    print(f"Scanning components directory: {components_dir}")
    mapping = find_component_classes(str(components_dir))
    
    # Save to JSON file
    output_file = script_dir / "components.json"
    with open(output_file, 'w') as f:
        json.dump(mapping, f, indent=2, sort_keys=True)
    
    print(f"\n✅ Created component mapping: {output_file}")
    print(f"   Found {len(mapping)} components")
    
    return 0


if __name__ == "__main__":
    exit(main())

