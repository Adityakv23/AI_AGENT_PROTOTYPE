# Quick Start Guide

## 🚀 Getting Started in 5 Minutes

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Prepare Data

```bash
python cli.py prepare-data
```

This creates:
- `data/training_data.json` - Training dataset
- `data/test_data.json` - Test dataset

### Step 3: (Optional) Fine-Tune Model

For best results, fine-tune the model:

```bash
python cli.py train
```

**Note**: Training takes ~10-15 minutes on a GPU. Skip this to use base model.

### Step 4: Setup RAG

Initialize the knowledge base:

```bash
python cli.py setup-rag
```

### Step 5: Try It Out!

#### Interactive Mode (Recommended)
```bash
python cli.py interactive
```

#### Single Summarization
```bash
python cli.py test-summarize --text "Your academic paper abstract here..."
```

#### Run Full Demo
```bash
python cli.py demo
```

## 📝 Example Usage

### Basic Summarization

```bash
python cli.py test-summarize -t "This paper presents a novel deep learning approach..."
```

### From File

```bash
python cli.py test-summarize -f paper.txt
```

### Check System Status

```bash
python cli.py status
```

### Run Evaluation

```bash
python cli.py evaluate
```

## 🎯 Using the Python API

```python
from agents import AIAgent

# Initialize agent
agent = AIAgent(use_finetuned=True)  # Set to False to use base model

# Summarize a paper
paper_text = "Your academic paper text here..."
summary = agent.summarize_academic_paper(paper_text)

print(summary)
```

## 📊 Understanding Outputs

### Summary Output
- **Concise**: <20% of original length
- **Academic style**: Formal terminology
- **Key points**: Main contributions and findings

### Evaluation Report
Located in `output/evaluation_report.json`:
- ROUGE scores
- BERTScore
- Compression ratio
- Quality metrics

## 🔧 Troubleshooting

### Issue: "Fine-tuned model not found"
**Solution**: Run `python cli.py train` or use base model with `--use-base` flag

### Issue: CUDA out of memory
**Solution**: Model automatically uses CPU if GPU unavailable

### Issue: Import errors
**Solution**: Ensure all dependencies installed: `pip install -r requirements.txt`

## 📚 Next Steps

1. Read [ARCHITECTURE.md](ARCHITECTURE.md) for system details
2. Explore [README.md](README.md) for full documentation
3. Customize in `config.py`
4. Add your own data to `data/training_data.json`

## 🆘 Need Help?

Check the documentation or run:
```bash
python cli.py --help
python cli.py <command> --help
```

---

**Enjoy automating your academic paper summarization! 🎓**

