"""
Configuration settings for the AI Agent Prototype
"""

import os
from pathlib import Path
from typing import Optional

# Base paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "output"
LOGS_DIR = BASE_DIR / "logs"

# Create directories
for directory in [DATA_DIR, MODELS_DIR, OUTPUT_DIR, LOGS_DIR]:
    directory.mkdir(exist_ok=True)

# Model configuration
BASE_MODEL_NAME = "distilgpt2"  # Will be fine-tuned
BASE_MODEL_SIZE = "medium"  # Can be small, medium, or large

# Fine-tuning configuration
FINE_TUNING_CONFIG = {
    "method": "lora",  # LoRA (Parameter-Efficient Fine-Tuning)
    "r": 8,  # LoRA rank
    "lora_alpha": 16,
    "lora_dropout": 0.1,
    "target_modules": ["c_attn", "c_proj"],  # For GPT-based models
}

# Training configuration
TRAINING_CONFIG = {
    "batch_size": 4,
    "gradient_accumulation_steps": 4,
    "learning_rate": 5e-4,
    "num_epochs": 3,
    "warmup_steps": 100,
    "max_length": 512,
    "save_steps": 500,
    "eval_steps": 100,
}

# RAG configuration
RAG_CONFIG = {
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "vector_store_path": DATA_DIR / "vector_store",
    "top_k": 5,
    "chunk_size": 512,
    "chunk_overlap": 50,
}

# Evaluation configuration
EVALUATION_CONFIG = {
    "metrics": ["rouge", "bert_score", "semantic_similarity"],
    "rouge_metrics": ["rouge1", "rouge2", "rougeL"],
}

# Agent configuration
AGENT_CONFIG = {
    "max_iterations": 10,
    "max_plan_length": 20,
    "temperature": 0.7,
    "max_tokens": 1000,
}

# File paths
TRAINING_DATA_PATH = DATA_DIR / "training_data.json"
TEST_DATA_PATH = DATA_DIR / "test_data.json"
FINE_TUNED_MODEL_PATH = MODELS_DIR / "fine_tuned_model"
VECTOR_STORE_PATH = DATA_DIR / "vector_store"

# API keys (load from environment)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "")

# User information (to be filled)
USER_INFO = {
    "name": "Aditya Kumar",
    "university": "Your University",
    "department": "Your Department",
}


