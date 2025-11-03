# AI Agent Prototype - Academic Paper Summarization System

**Author:** Aditya Kumar  
**Institution:** IIT KANPUR  
**Department:** Electrical Engineering

## 📋 Project Overview

This project implements an AI agent prototype that automates the task of summarizing academic research papers. The agent uses advanced AI techniques including fine-tuned models, multi-agent collaboration, RAG (Retrieval-Augmented Generation), and comprehensive evaluation metrics.

### **Core Task**
The manual task being automated is **academic paper summarization** - a time-consuming but critical task for researchers, students, and academics who need to quickly understand and digest large volumes of research literature.

## 🎯 Core Features (Mandatory)

### 1. Fine-Tuned Model with LoRA ✅

**Why LoRA?**
- **Parameter Efficiency**: Trains less than 1% of total parameters versus full fine-tuning
- **Task Specialization**: Adapts the base model specifically for academic summarization style
- **Improved Reliability**: Produces consistent, concise summaries with proper academic terminology
- **Adapted Style**: Learns to generate summaries in the formal, structured style expected in academia
- **Speed**: Faster training and inference compared to full fine-tuning

**Implementation:**
- Base Model: DistilGPT2 (lightweight, efficient)
- Fine-tuning Method: Parameter-Efficient Fine-Tuning (PEFT) with LoRA
- LoRA Configuration:
  - Rank (r): 8
  - Alpha: 16
  - Target Modules: attention layers (c_attn, c_proj, q_attn)
  - Dropout: 0.1

### 2. Evaluation Metrics ✅

Comprehensive evaluation system measuring:
- **ROUGE Scores**: N-gram overlap (ROUGE-1, ROUGE-2, ROUGE-L)
- **BERTScore**: Semantic similarity using contextual embeddings
- **Compression Ratio**: Summary length relative to original text
- **Quality Scoring**: Task-specific metrics for academic summaries

## 🚀 Optional Features (Bonus Points)

### 1. Multi-Agent Collaboration ✅

**Architecture:**
- **Planner Agent**: Breaks down tasks into actionable sequential steps
- **Executor Agent**: Carries out planned actions with RAG-enhanced context
- **Coordinator**: Main AI Agent managing agent interactions

**Benefits:**
- Modular design for complex reasoning
- Iterative refinement based on feedback
- Specialized agents for different capabilities

### 2. RAG Integration ✅

**Retrieval-Augmented Generation** for enhanced document understanding:
- Vector database (ChromaDB) with semantic embeddings
- Sentence transformers for chunking and retrieval
- Context augmentation for improved summarization
- Knowledge base built from academic literature

### 3. User Interface ✅

**CLI (Command-Line Interface)** with rich output:
- Interactive mode for real-time task processing
- Batch processing for multiple documents
- Status monitoring and configuration management
- Evaluation and benchmarking tools

## 🏗️ Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                      AI Agent Prototype                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────────┐         ┌──────────────────┐          │
│  │  Planner Agent  │────────▶│  Executor Agent  │          │
│  │   (Reasoning)   │         │   (Execution)    │          │
│  └─────────────────┘         └──────────────────┘          │
│         │                              │                    │
│         └────────────┬─────────────────┘                    │
│                      ▼                                      │
│              ┌──────────────────┐                           │
│              │  Fine-tuned LoRA │                           │
│              │  Model (DistilGPT2)│                         │
│              └──────────────────┘                           │
│                      │                                      │
│         ┌────────────┴────────────┐                         │
│         ▼                         ▼                         │
│  ┌─────────────┐          ┌─────────────┐                  │
│  │  RAG System │          │  Evaluation │                  │
│  │  (ChromaDB) │          │   Metrics   │                  │
│  └─────────────┘          └─────────────┘                  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Interaction Flow

1. **Input Processing**: User provides academic paper or query
2. **Planning**: Planner Agent breaks down into steps
3. **Context Retrieval**: RAG system augments with relevant information
4. **Execution**: Executor Agent generates summaries using fine-tuned model
5. **Evaluation**: Quality metrics computed
6. **Output**: Structured summary delivered

## 📂 Project Structure

```
i'mbesidesyou/
├──DATA_SCIENCE_REPORT/
|   └── ARCHITECTURE.md  # architecture of the AI agent
|   └── README.md  # README file
|   └── QUICKSTART.md  # quick guide on how to use the agent
|   └── PROJECT_SUMMARY.md  # Detailed summary of the AI agent project
|   └── SUBMISSION.md  # final summary 
|
├──SOURCE_CODE/
|   └── config.py  #Configuration management
|   └── data_preparation.py  # dataset creation
|   └── fine_tuning.py  # LoRA fine tuning
|   └── agents.py  # Multi-agent system
|   └── rag_system.py  # RAG implementataion
|   └── evaluation.py  # Evaluation metrics
|   └── cli.py  # command line interface
│
├── UTILITIES/                    # utilities for the agent
│   ├── requirements.txt    # All resources which would be needed for the agent
│   ├── setup.py     # tests and trains data
│   └── test_basic   # basic test dataset
|
├── data/                    # Data directory
│   ├── training_data.json  # Training dataset
│   ├── test_data.json      # Test dataset
│   └── vector_store/       # RAG vector database
│
├── models/                  # Model directory
│   └── fine_tuned_model/   # Fine-tuned LoRA model
|
├── logs/               # Output directory
|    └── logs.txt  # prompts used and responses given by the agent
|    └── 1,2,3.jpg  # screenshot of the complete log 
│
└── output/                  # Output directory
    └── evaluation_report.json
```

## 🛠️ Installation & Setup

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- 8GB+ RAM

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd i'mbesidesyou
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Prepare datasets**
```bash
python cli.py prepare-data
```

4. **Fine-tune the model** (optional but recommended)
```bash
python cli.py train
```

5. **Initialize RAG system**
```bash
python cli.py setup-rag
```

## 🎮 Usage

### Interactive Mode

```bash
python cli.py interactive
```

### Single Summarization

```bash
python cli.py test-summarize --text "Your academic paper text here..."
```

### From File

```bash
python cli.py test-summarize --file paper.txt
```

### Evaluation

```bash
python cli.py evaluate
```

### System Status

```bash
python cli.py status
```

### Demo

```bash
python cli.py demo
```

## 📊 Evaluation Results

### Metrics Overview

- **ROUGE-1 F1**: Measures unigram overlap with reference summaries
- **ROUGE-2 F1**: Measures bigram overlap for fluency
- **ROUGE-L F1**: Measures longest common subsequence
- **BERTScore**: Semantic similarity using contextual embeddings
- **Compression Ratio**: Summary efficiency

### Why These Metrics?

- **ROUGE**: Industry standard for summarization evaluation
- **BERTScore**: Captures semantic meaning beyond n-grams
- **Compression Ratio**: Ensures summaries are concise (goal: <20% of original)

## 🎓 Academic Paper Summarization: Data Science Report

### Fine-Tuning Setup

**Method**: Parameter-Efficient Fine-Tuning (LoRA)  
**Base Model**: DistilGPT2 (82M parameters)  
**Training Data**: 24 academic paper samples with summaries  
**Test Split**: 20% (5 samples)

**Hyperparameters**:
- Learning Rate: 5e-4
- Batch Size: 4 (with gradient accumulation ×4)
- Epochs: 3
- LoRA Rank: 8
- LoRA Alpha: 16

**Why This Configuration**:
- DistilGPT2 provides good balance of performance and efficiency
- LoRA reduces trainable parameters from 82M to <500K (<1%)
- Task-specific adaptation for academic language and structure

### Evaluation Methodology

**Quantitative**:
- Automated metrics (ROUGE, BERTScore)
- Compression ratio analysis
- Quality score computation

**Qualitative**:
- Manual review of summary coherence
- Academic terminology usage
- Factual accuracy verification

### Outcomes

*Note: Full evaluation results will be generated after training*

## 🔧 Configuration

Edit `config.py` to customize:
- Model parameters
- Training hyperparameters
- RAG settings
- Evaluation metrics

## 🤝 References

- [Building Effective Agents (Anthropic)](https://www.anthropic.com/engineering/building-effective-agents)
- [AI Agent Design Pattern Paper](https://arxiv.org/pdf/2405.10467)
- Hugging Face Transformers
- PEFT Library
- RAG Research Papers

## 📄 License

This project is developed for academic purposes as part of the Data Science Internship Assignment.

---

**Developed with ❤️ for automating academic research workflows**

