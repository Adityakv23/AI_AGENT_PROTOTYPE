"""
Setup script for AI Agent Prototype
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: {e.stderr}")
        return False


def main():
    print("\n" + "="*60)
    print("AI Agent Prototype - Setup")
    print("="*60)
    
    steps = [
        ("python cli.py prepare-data", "Preparing datasets"),
        ("python cli.py setup-rag", "Initializing RAG system"),
    ]
    
    # Check if training is needed
    model_path = Path("models/fine_tuned_model")
    if not model_path.exists():
        print("\n[INFO] Fine-tuned model not found.")
        train = input("Would you like to train the model now? (y/n): ").lower().strip()
        if train == 'y':
            steps.append(("python cli.py train", "Training fine-tuned model"))
    
    # Execute setup steps
    for cmd, desc in steps:
        success = run_command(cmd, desc)
        if not success:
            print(f"\n❌ Setup failed at: {desc}")
            sys.exit(1)
    
    # Final status
    print("\n" + "="*60)
    print("Setup Complete!")
    print("="*60)
    print("\nNext steps:")
    print("  1. Run: python cli.py interactive")
    print("  2. Or: python cli.py test-summarize --text \"Your paper...\"")
    print("  3. Read: QUICKSTART.md for more examples")
    print()


if __name__ == "__main__":
    main()

