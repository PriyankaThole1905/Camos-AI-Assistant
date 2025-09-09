# src/utils.py
import yaml
import time
import requests # Keeping for Ollama client, though LangChain wrappers are used for core LLM
import json
import os

def load_config(filepath):
    """Loads a YAML configuration file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Config file not found: {filepath}")
    with open(filepath, 'r') as f:
        return yaml.safe_load(f)

def load_prompt_templates(filepath):
    """Loads prompt templates from a YAML file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Prompt templates file not found: {filepath}")
    with open(filepath, 'r') as f:
        return yaml.safe_load(f)

# The exponential_backoff_request is less critical now as LangChain's Ollama integration
# handles retries and connection, but can be kept if direct requests are still needed.
# For this LangChain version, we'll remove it to simplify.