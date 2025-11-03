"""
Fine-tuning module for creating a parameter-efficient LoRA model
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
import json
from pathlib import Path
from typing import List, Dict, Any

from config import (
    FINE_TUNED_MODEL_PATH,
    TRAINING_CONFIG,
    FINE_TUNING_CONFIG,
    TRAINING_DATA_PATH,
    TEST_DATA_PATH
)


class FineTuner:
    """
    Fine-tune a language model using LoRA for academic paper summarization
    """
    
    def __init__(self, base_model_name: str = "distilgpt2"):
        self.base_model_name = base_model_name
        self.model = None
        self.tokenizer = None
        self.training_config = TRAINING_CONFIG
        self.lora_config = FINE_TUNING_CONFIG
        
        # Note: Using DistilGPT2 instead of DialoGPT for better compatibility
        # This is a smaller, more efficient model that works well with LoRA
        print(f"Initializing fine-tuner with base model: {base_model_name}")
        print(f"Reason: DistilGPT2 is smaller, more efficient, and perfect for LoRA fine-tuning")
    
    def load_model_and_tokenizer(self):
        """Load the base model and tokenizer"""
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            dtype=torch.float16,
            device_map="auto"
        )
        
        # Prepare model for LoRA
        self.model = prepare_model_for_kbit_training(self.model)
        
    def setup_lora(self):
        """Configure and apply LoRA to the model"""
        # For GPT-2 based models, adjust target modules
        lora_config = LoraConfig(
            r=self.lora_config["r"],
            lora_alpha=self.lora_config["lora_alpha"],
            target_modules=["c_attn", "c_proj", "q_attn"],  # GPT-2 layers
            lora_dropout=self.lora_config["lora_dropout"],
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        
        print("✓ LoRA configuration applied")
        print(f"  - Rank (r): {lora_config.r}")
        print(f"  - Alpha: {lora_config.lora_alpha}")
        print(f"  - Target modules: {lora_config.target_modules}")
        print("\nWhy LoRA?")
        print("  1. Parameter Efficiency: Trains <1% of parameters vs full fine-tuning")
        print("  2. Task Specialization: Adapts model for academic summarization")
        print("  3. Reliability: Produces consistent, concise academic summaries")
        print("  4. Speed: Faster training and inference")
    
    def prepare_dataset(self, data_path: Path) -> Dataset:
        """Load and prepare dataset for training"""
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        def format_prompt(example):
            # Create a prompt for the summarization task
            input_text = f"Summarize: {example['input']}\nSummary:"
            full_text = input_text + " " + example['output'] + self.tokenizer.eos_token
            return {"text": full_text}
        
        dataset = [format_prompt(item) for item in data]
        return Dataset.from_list(dataset)
    
    def tokenize_dataset(self, dataset: Dataset) -> Dataset:
        """Tokenize the dataset"""
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=self.training_config["max_length"],
                padding="max_length"
            )
        
        tokenized = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"]
        )
        return tokenized
    
    def train(self) -> None:
        """Train the fine-tuned model"""
        print("\n=== Starting Fine-Tuning Process ===\n")
        
        # Load model and setup LoRA
        self.load_model_and_tokenizer()
        self.setup_lora()
        
        # Prepare datasets
        print("\n--- Preparing Datasets ---")
        train_dataset = self.prepare_dataset(TRAINING_DATA_PATH)
        eval_dataset = self.prepare_dataset(TEST_DATA_PATH)
        
        train_tokenized = self.tokenize_dataset(train_dataset)
        eval_tokenized = self.tokenize_dataset(eval_dataset)
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # Training arguments
        # Handle both old and new API
        try:
            # Try new API first
            training_args = TrainingArguments(
                output_dir=str(FINE_TUNED_MODEL_PATH),
                overwrite_output_dir=True,
                num_train_epochs=self.training_config["num_epochs"],
                per_device_train_batch_size=self.training_config["batch_size"],
                gradient_accumulation_steps=self.training_config["gradient_accumulation_steps"],
                learning_rate=self.training_config["learning_rate"],
                warmup_steps=self.training_config["warmup_steps"],
                logging_steps=50,
                save_steps=self.training_config["save_steps"],
                eval_strategy="steps",  # New API
                eval_steps=self.training_config["eval_steps"],
                save_total_limit=3,
                load_best_model_at_end=True,
                metric_for_best_model="loss",
                greater_is_better=False,
                fp16=True,
                report_to="none",
            )
        except TypeError:
            # Fall back to old API
            training_args = TrainingArguments(
                output_dir=str(FINE_TUNED_MODEL_PATH),
                overwrite_output_dir=True,
                num_train_epochs=self.training_config["num_epochs"],
                per_device_train_batch_size=self.training_config["batch_size"],
                gradient_accumulation_steps=self.training_config["gradient_accumulation_steps"],
                learning_rate=self.training_config["learning_rate"],
                warmup_steps=self.training_config["warmup_steps"],
                logging_steps=50,
                save_steps=self.training_config["save_steps"],
                evaluation_strategy="steps",  # Old API
                eval_steps=self.training_config["eval_steps"],
                save_total_limit=3,
                load_best_model_at_end=True,
                metric_for_best_model="loss",
                greater_is_better=False,
                fp16=True,
                report_to="none",
            )
        
        # Initialize Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_tokenized,
            eval_dataset=eval_tokenized,
            data_collator=data_collator,
        )
        
        # Train
        print("\n--- Training Model ---")
        trainer.train()
        
        # Save model
        print("\n--- Saving Model ---")
        self.model.save_pretrained(str(FINE_TUNED_MODEL_PATH))
        self.tokenizer.save_pretrained(str(FINE_TUNED_MODEL_PATH))
        
        print(f"\n✓ Fine-tuning complete! Model saved to: {FINE_TUNED_MODEL_PATH}")
    
    def load_fine_tuned_model(self):
        """Load the fine-tuned model"""
        if not FINE_TUNED_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Fine-tuned model not found at {FINE_TUNED_MODEL_PATH}. "
                "Please run train() first."
            )
        
        self.tokenizer = AutoTokenizer.from_pretrained(str(FINE_TUNED_MODEL_PATH))
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            str(FINE_TUNED_MODEL_PATH),
            dtype=torch.float16,
            device_map="auto"
        )
        
        print(f"✓ Loaded fine-tuned model from {FINE_TUNED_MODEL_PATH}")


if __name__ == "__main__":
    # Initialize and train
    fine_tuner = FineTuner(base_model_name="distilgpt2")
    fine_tuner.train()

