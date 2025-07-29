#!/usr/bin/env python3
"""Check which file is still importing newspaper"""

import os
import re


def find_newspaper_imports(directory):
    """Find all files that import newspaper"""
    pattern = re.compile(r'from\s+newspaper\s+import|import\s+newspaper')

    for root, dirs, files in os.walk(directory):
        # Skip venv and __pycache__
        dirs[:] = [d for d in dirs if d not in ['venv', '__pycache__', '.git']]

        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'r') as f:
                        content = f.read()
                        if pattern.search(content):
                            print(f"Found newspaper import in: {filepath}")
                            # Show the line
                            for i, line in enumerate(content.splitlines()):
                                if pattern.search(line):
                                    print(f"  Line {i + 1}: {line.strip()}")
                except Exception as e:
                    print(f"Error reading {filepath}: {e}")


if __name__ == "__main__":
    print("Searching for newspaper imports...")
    print("=" * 60)
    find_newspaper_imports(".")