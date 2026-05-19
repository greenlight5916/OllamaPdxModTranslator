#!/usr/bin/env python3
"""Entry point for OllamaTranslator."""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

if __name__ == "__main__":
    from ollama_translator.run_ollama_translator import main
    main()
