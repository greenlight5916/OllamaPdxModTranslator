#!/usr/bin/env python3
import sys, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
from ollama_translator_app import main

if __name__ == "__main__":
    main()
