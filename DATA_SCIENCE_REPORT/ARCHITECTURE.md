# AI Agent Architecture Document

## Overview

This document describes the architecture, design decisions, and implementation details of the Academic Paper Summarization AI Agent.

## Executive Summary

The system is designed as a modular, multi-agent AI system that automates academic paper summarization through:
- **Fine-tuned Language Models**: Using LoRA for parameter-efficient adaptation
- **Multi-Agent Collaboration**: Planner and Executor agents working together
- **RAG Integration**: Context augmentation through retrieval
- **Comprehensive Evaluation**: Multiple metrics for quality assurance

## Core Components

### 1. Fine-Tuned Model Layer

**Component**: `fine_tuning.py`

**Purpose**: Adapt a base language model for academic summarization tasks

**Key Design Decisions**:
1. **Base Model Choice**: DistilGPT2
   - **Rationale**: Balance between capability and efficiency
   - Lightweight (82M parameters) for faster training/inference
   - Good performance for text generation tasks
   - Compatible with LoRA fine-tuning

2. **Fine-Tuning Method**: LoRA (Low-Rank Adaptation)
   - **Why LoRA?**:
     - Trains <1% of parameters (82M → ~500K)
     - Prevents catastrophic forgetting
     - Faster training cycles
     - Lower resource requirements
   - **Configuration**:
     - Rank (r) = 8: Good balance of expressivity and efficiency
     - Alpha = 16: Scaling factor for learned weights
     - Target modules: Attention layers (c_attn, c_proj, q_attn)

3. **Training Strategy**:
   - Small batch size (4) with gradient accumulation
   - Learning rate: 5e-4 (conservative for stability)
   - Mixed precision (FP16) for efficiency
   - Early stopping based on validation loss

### 2. Multi-Agent System

**Component**: `agents.py`

**Architecture**:
```
┌─────────────────────────────────┐
│      AI Agent (Coordinator)     │
└─────────────────────────────────┘
              │
      ┌───────┴───────┐
      │               │
┌──────────────┐ ┌──────────────┐
│   Planner    │ │  Executor    │
│   Agent      │ │  Agent       │
└──────────────┘ └──────────────┘
     │                │
     │                │
     ▼                ▼
 Generate         Execute
   Plan          Steps with
                    RAG
```

**Planner Agent**:
- **Role**: Break down high-level tasks into actionable steps
- **Capabilities**:
  - Task decomposition
  - Reasoning explanation
  - Confidence estimation
  - Plan refinement based on feedback
- **Model**: Fine-tuned DistilGPT2

**Executor Agent**:
- **Role**: Execute individual steps with context
- **Capabilities**:
  - Step-by-step execution
  - RAG-enhanced context retrieval
  - Quality self-assessment
  - Result compilation
- **Integration**: Direct access to RAG system

**Benefits of Multi-Agent Design**:
1. **Modularity**: Each agent has specialized responsibility
2. **Scalability**: Easy to add new agent types
3. **Debugging**: Can isolate issues to specific agents
4. **Flexibility**: Agents can be swapped or improved independently

### 3. RAG System

**Component**: `rag_system.py`

**Purpose**: Provide contextual information through retrieval augmentation

**Implementation**:
- **Vector Database**: ChromaDB
- **Embedding Model**: sentence-transformers/all-MiniLM-L6-v2
- **Chunking**: 512 tokens with 50 token overlap
- **Retrieval**: Top-K (5) semantic similarity

**Why RAG?**:
- Enhances factual accuracy
- Provides domain-specific context
- Reduces hallucinations
- Enables knowledge base updates without retraining

**Workflow**:
1. Document ingestion → Chunking → Embedding
2. Query → Embedding → Similarity search
3. Retrieved chunks → Context augmentation
4. Augmented prompt → Model generation

### 4. Evaluation System

**Component**: `evaluation.py`

**Metrics**:

1. **ROUGE (Recall-Oriented Understudy for Gisting Evaluation)**:
   - ROUGE-1: Unigram overlap
   - ROUGE-2: Bigram overlap
   - ROUGE-L: Longest common subsequence
   - **Why**: Industry standard for summarization

2. **BERTScore**:
   - Contextual embedding similarity
   - **Why**: Captures semantic meaning beyond n-grams

3. **Compression Ratio**:
   - Summary length / Original length
   - **Why**: Ensures conciseness (target: <20%)

4. **Quality Scores**:
   - Completeness
   - Coherence
   - Factual accuracy (manual)

### 5. Data Preparation

**Component**: `data_preparation.py`

**Dataset**:
- 8 academic paper abstracts with summaries
- Expanded to 24 samples with variations
- Train/Test split: 80/20

**Why This Dataset**:
- Simulates real academic summarization task
- Covers multiple research domains
- Includes proper academic terminology
- Demonstrates model capability

## Interaction Flow

### High-Level Flow

```
User Input (Paper Text)
        │
        ▼
┌──────────────────────┐
│  AI Agent: Process   │
└──────────────────────┘
        │
        ├──► Planner: Generate Plan
        │         │
        │         ▼
        │    ┌──────────────┐
        │    │ Plan Steps   │
        │    └──────────────┘
        │         │
        │         ▼
        ├──► Executor: Execute Each Step
        │         │
        │         ├──► RAG: Retrieve Context
        │         │
        │         ├──► Fine-tuned Model: Generate
        │         │
        │         └──► Self-Assess Quality
        │
        ▼
┌──────────────────────┐
│  Compile Results     │
└──────────────────────┘
        │
        ▼
┌──────────────────────┐
│  Evaluation Metrics  │
└──────────────────────┘
        │
        ▼
User Output (Summary)
```

### Detailed Step-by-Step

1. **Input Processing**
   - User provides academic paper text
   - System validates and preprocesses input

2. **Planning Phase**
   - Planner agent receives task
   - Generates structured plan with steps
   - Returns: steps, reasoning, confidence

3. **Execution Phase** (for each step)
   - Executor receives step
   - RAG retrieves relevant context (if applicable)
   - Fine-tuned model generates output
   - Quality assessment performed

4. **Compilation**
   - Combine step outputs
   - Format final summary

5. **Evaluation**
   - Compute metrics
   - Generate report

## Technology Stack

### Core
- **PyTorch**: Deep learning framework
- **Transformers (Hugging Face)**: Model library
- **PEFT**: Parameter-efficient fine-tuning

### RAG & Embeddings
- **ChromaDB**: Vector database
- **Sentence Transformers**: Embedding models
- **LangChain**: Orchestration framework

### Evaluation
- **Rouge-Score**: ROUGE metrics
- **BERTScore**: Semantic similarity

### UI/CLI
- **Typer**: CLI framework
- **Rich**: Terminal formatting

### Utilities
- **PyPDF**: PDF processing
- **NumPy/Pandas**: Data handling

## Configuration Management

All configuration centralized in `config.py`:
- Model hyperparameters
- Training settings
- RAG configuration
- Evaluation metrics
- File paths

**Benefits**:
- Single source of truth
- Easy experimentation
- Environment-specific settings
- Version control friendly

## Data Flow

### Training Pipeline
```
Raw Academic Papers
       │
       ▼
Data Preparation (JSON format)
       │
       ▼
Fine-Tuning (LoRA)
       │
       ▼
Fine-Tuned Model
       │
       ├──► Evaluation
       └──► Deployment
```

### Inference Pipeline
```
User Query
       │
       ▼
Planner (Task → Plan)
       │
       ▼
Executor (Plan → Results)
       │
       ├──► RAG (Context Retrieval)
       └──► Fine-Tuned Model (Generation)
       │
       ▼
Evaluation (Quality Check)
       │
       ▼
Summary Output
```

## Scalability Considerations

### Current Limitations
- Single GPU training
- Small dataset (24 samples)
- Sequential execution

### Future Improvements
1. **Distributed Training**: Multi-GPU LoRA fine-tuning
2. **Larger Dataset**: Thousands of academic papers
3. **Parallel Execution**: Concurrent agent processing
4. **Caching**: RAG query result caching
5. **Streaming**: Real-time generation

## Security & Privacy

- **No External APIs**: All processing local
- **Data Isolation**: Separate training/test sets
- **Model Isolation**: Fine-tuned model stays local
- **No User Data Storage**: Input not persistently stored

## Error Handling

- **Graceful Degradation**: Falls back to base model if fine-tuned unavailable
- **Input Validation**: Checks for empty/invalid inputs
- **Exception Handling**: Try-catch blocks throughout
- **Logging**: Structured logging for debugging

## Testing Strategy

1. **Unit Tests**: Individual component testing
2. **Integration Tests**: Multi-agent workflows
3. **Evaluation Tests**: Metric computation
4. **End-to-End Tests**: Full pipeline

## Deployment Considerations

### Development Environment
- Single machine
- Local GPU
- Python environment

### Production Readiness
- Containerization (Docker)
- API server (FastAPI)
- Model serving
- Monitoring & logging

## References

1. **LoRA Paper**: "LoRA: Low-Rank Adaptation of Large Language Models"
2. **RAG Paper**: "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
3. **Multi-Agent Systems**: "Building Effective Agents" (Anthropic)
4. **ROUGE Paper**: "ROUGE: A Package for Automatic Evaluation of Summaries"

---

**Document Version**: 1.0  
**Last Updated**: [Current Date]

