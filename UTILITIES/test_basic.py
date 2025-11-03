"""
Basic test script to verify core functionality without external dependencies
"""

def test_imports():
    """Test if required modules can be imported"""
    print("Testing imports...")
    
    try:
        import json
        import pathlib
        print("✓ Core Python libraries OK")
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    
    # Try optional dependencies
    optional_imports = [
        ('torch', 'PyTorch'),
        ('transformers', 'Transformers'),
        ('typer', 'Typer'),
        ('rich', 'Rich'),
        ('chromadb', 'ChromaDB'),
        ('rouge_score', 'Rouge Score'),
        ('bert_score', 'BERT Score'),
        ('sentence_transformers', 'Sentence Transformers'),
    ]
    
    print("\nOptional dependencies:")
    missing = []
    for module, name in optional_imports:
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} (not installed)")
            missing.append(name)
    
    if missing:
        print(f"\n⚠ Missing dependencies. Run: pip install -r requirements.txt")
        print(f"Missing: {', '.join(missing)}")
    else:
        print("\n✓ All dependencies installed!")
    
    return len(missing) == 0


def test_config():
    """Test configuration module"""
    print("\nTesting configuration...")
    try:
        from config import (
            DATA_DIR, MODELS_DIR, OUTPUT_DIR,
            TRAINING_CONFIG, FINE_TUNING_CONFIG,
            RAG_CONFIG, EVALUATION_CONFIG
        )
        print("✓ Config module OK")
        print(f"  Data directory: {DATA_DIR}")
        print(f"  Models directory: {MODELS_DIR}")
        print(f"  Output directory: {OUTPUT_DIR}")
        return True
    except Exception as e:
        print(f"✗ Config error: {e}")
        return False


def test_data_preparation():
    """Test data preparation without generating files"""
    print("\nTesting data preparation...")
    try:
        from data_preparation import DataPreparator
        preparator = DataPreparator()
        sample_data = preparator.create_sample_academic_dataset()
        print(f"✓ Data preparation OK ({len(sample_data)} samples)")
        return True
    except Exception as e:
        print(f"✗ Data preparation error: {e}")
        return False


def test_structure():
    """Check project structure"""
    print("\nChecking project structure...")
    from pathlib import Path
    
    required_files = [
        'README.md',
        'ARCHITECTURE.md',
        'QUICKSTART.md',
        'requirements.txt',
        'config.py',
        'data_preparation.py',
        'fine_tuning.py',
        'agents.py',
        'rag_system.py',
        'evaluation.py',
        'cli.py',
        'setup.py',
    ]
    
    missing_files = []
    for file in required_files:
        if Path(file).exists():
            print(f"  ✓ {file}")
        else:
            print(f"  ✗ {file}")
            missing_files.append(file)
    
    if missing_files:
        print(f"\n⚠ Missing files: {missing_files}")
        return False
    else:
        print("\n✓ All required files present!")
        return True


def main():
    """Run all tests"""
    print("="*60)
    print("AI Agent Prototype - Basic Tests")
    print("="*60)
    
    tests = [
        ("Imports", test_imports),
        ("Configuration", test_config),
        ("Data Preparation", test_data_preparation),
        ("Project Structure", test_structure),
    ]
    
    results = []
    for name, test_func in tests:
        result = test_func()
        results.append((name, result))
    
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{name}: {status}")
    
    all_passed = all(result for _, result in results)
    
    if all_passed:
        print("\n🎉 All tests passed!")
    else:
        print("\n⚠️  Some tests failed. See details above.")
        print("\nTo fix dependency issues, run:")
        print("  pip install -r requirements.txt")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()

