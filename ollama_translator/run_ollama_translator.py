#!/usr/bin/env python3
import sys, os

# Add parent directory (project root) to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from ollama_translator.ollama_translator_app import main

if __name__ == "__main__":
    main()
