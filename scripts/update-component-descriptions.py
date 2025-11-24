#!/usr/bin/env python3
"""
Script to update component descriptions in node.json by extracting them from the actual component files.
This ensures that node.json contains the accurate, up-to-date descriptions from the source code.
"""

import json
import re
import os
from pathlib import Path


def find_component_file(component_path: str) -> Path:
    """Convert dot notation path to file path."""
    file_path = component_path.replace(".", "/") + ".py"
    return Path(file_path)


def extract_component_info(file_path: Path, component_name: str) -> tuple:
    """Extract display name and description from component file.

    Returns:
        tuple: (display_name, description)
    """
    if not file_path.exists():
        return None, None

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Remove Component suffix if present for class matching
        base_name = component_name.replace("Component", "")

        # Look for the main component class - more comprehensive patterns
        class_patterns = [
            # Pattern 1: class Name(Component): ...
            r"class\s+" + re.escape(component_name) + r"\s*\([^)]*\):(.*?)(?=class\s+|\Z)",
            r"class\s+" + re.escape(base_name) + r"\s*\([^)]*\):(.*?)(?=class\s+|\Z)",
            # Pattern 2: class Name: ...
            r"class\s+" + re.escape(component_name) + r"\s*:(.*?)(?=class\s+|\Z)",
            r"class\s+" + re.escape(base_name) + r"\s*:(.*?)(?=class\s+|\Z)",
            # Pattern 3: Component class patterns
            r"class\s+" + re.escape(base_name) + r"Component\s*\([^)]*\):(.*?)(?=class\s+|\Z)",
        ]

        for pattern in class_patterns:
            match = re.search(pattern, content, re.DOTALL | re.MULTILINE)
            if match:
                class_content = match.group(1)

                # Extract display name
                display_name = None
                display_patterns = [
                    r"display_name\s*[:=]\s*[\"']([^\"']+)[\"']",
                    r"display_name\s*:\s*str\s*[:=]\s*[\"']([^\"']+)[\"']",
                ]
                for dp in display_patterns:
                    dm = re.search(dp, class_content)
                    if dm:
                        display_name = dm.group(1).strip()
                        break

                # Extract description
                description = None
                desc_patterns = [
                    r"description\s*[:=]\s*[\"']([^\"']+)[\"']",
                    r"description\s*:\s*str\s*[:=]\s*[\"']([^\"']+)[\"']",
                ]
                for dp in desc_patterns:
                    dm = re.search(dp, class_content)
                    if dm:
                        description = dm.group(1).strip()
                        break

                if display_name or description:
                    return display_name, description

        # Fallback: look for any class with display_name and description
        class_blocks = re.findall(r"class\s+\w+[^:]*:(.*?)(?=class\s+\w+[^:]*:|\Z)", content, re.DOTALL)

        for block in class_blocks:
            display_name = None
            description = None

            # Extract display name
            display_match = re.search(r"display_name\s*[:=]\s*[\"']([^\"']+)[\"']", block)
            if display_match:
                display_name = display_match.group(1).strip()

            # Extract description
            desc_match = re.search(r"description\s*[:=]\s*[\"']([^\"']+)[\"']", block)
            if desc_match:
                description = desc_match.group(1).strip()

            if display_name or description:
                return display_name, description

        # Final fallback: look for any display_name and description field in the entire file
        all_display_matches = re.findall(r"display_name\s*[:=]\s*[\"']([^\"']+)[\"']", content)
        all_desc_matches = re.findall(r"description\s*[:=]\s*[\"']([^\"']+)[\"']", content)

        display_name = all_display_matches[0].strip() if all_display_matches else None
        description = all_desc_matches[0].strip() if all_desc_matches else None

        return display_name, description

    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None, None


def main():
    """Main function to update node.json with component display names and descriptions."""
    print("Updating component display names and descriptions in node.json...")

    # Load node.json
    try:
        with open('node.json', 'r', encoding='utf-8') as f:
            node_data = json.load(f)
    except Exception as e:
        print(f"Error reading node.json: {e}")
        return 1

    components = node_data.get('components', {})
    updated_display_count = 0
    updated_desc_count = 0
    missing_count = 0

    print(f"Processing {len(components)} components...")

    for component_name, component_info in components.items():
        current_description = component_info.get('description', '')
        current_display_name = component_info.get('display_name', '')
        component_path = component_info.get('path', '')

        if not component_path:
            print(f"⚠️  {component_name}: No path specified")
            missing_count += 1
            continue

        # Find the component file
        file_path = find_component_file(component_path)

        # Extract display name and description from file
        extracted_display_name, extracted_description = extract_component_info(file_path, component_name)

        display_updated = False
        desc_updated = False

        # Update display name if it's different
        if extracted_display_name and extracted_display_name != current_display_name:
            component_info['display_name'] = extracted_display_name
            print(f"✓ {component_name}: Updated display name")
            print(f"    Old: '{current_display_name}'")
            print(f"    New: '{extracted_display_name}'")
            display_updated = True
            updated_display_count += 1

        # Update description if it's different
        if extracted_description and extracted_description != current_description:
            component_info['description'] = extracted_description
            print(f"✓ {component_name}: Updated description")
            print(f"    Old: {current_description[:50]}...")
            print(f"    New: {extracted_description[:50]}...")
            desc_updated = True
            updated_desc_count += 1

        if not display_updated and not desc_updated:
            if extracted_display_name and extracted_description:
                print(f"✓ {component_name}: Display name and description already correct")
            elif extracted_display_name:
                print(f"✓ {component_name}: Display name already correct")
            elif extracted_description:
                print(f"✓ {component_name}: Description already correct")
            else:
                print(f"⚠️  {component_name}: Could not extract info from {file_path}")
                missing_count += 1

    # Save updated node.json
    try:
        with open('node.json', 'w', encoding='utf-8') as f:
            json.dump(node_data, f, indent=2, ensure_ascii=False)
        print(f"\n✅ Updated node.json:")
        print(f"  - Updated display names: {updated_display_count}")
        print(f"  - Updated descriptions: {updated_desc_count}")
    except Exception as e:
        print(f"Error saving node.json: {e}")
        return 1

    if missing_count > 0:
        print(f"⚠️  Could not extract info for {missing_count} components")

    print(f"\nSummary:")
    print(f"  Total components: {len(components)}")
    print(f"  Updated display names: {updated_display_count}")
    print(f"  Updated descriptions: {updated_desc_count}")
    print(f"  Missing info: {missing_count}")
    print(f"  Already correct: {len(components) - updated_display_count - updated_desc_count - missing_count}")

    return 0


if __name__ == "__main__":
    exit(main())