# AI Literature Review & Learning Suite

A comprehensive collection of AI tools and educational tutorials for literature review automation, LLM training, and AI agent development.

## 📚 Table of Contents

- [Overview](#overview)
- [Components](#components)
- [Installation](#installation)
- [Literature Review System](#literature-review-system)
- [LLM Training Tutorial](#llm-training-tutorial)
- [AI Agent Tutorial](#ai-agent-tutorial)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

## 🎯 Overview

This project provides three main components:

1. **Literature Review System** (`lit_review.py`) - Automated PDF analysis and literature review generation
2. **LLM Training Tutorial** (`llm_training_tutorial.py`) - Educational guide for training language models from scratch
3. **AI Agent Tutorial** (`agent_from_scratch.py`) - Comprehensive tutorial on building AI agents

## 🧩 Components

### 1. Literature Review System (`lit_review.py`)
- **Purpose**: Automatically analyze PDF papers and generate comprehensive literature reviews
- **Features**: PDF text extraction, knowledge graph construction, concept analysis, automated summarization
- **Output**: Structured literature review in Markdown format

### 2. LLM Training Tutorial (`llm_training_tutorial.py`)
- **Purpose**: Educational guide for understanding and training language models
- **Features**: Transformer architecture implementation, tokenization, training loops, text generation
- **Learning**: Complete pipeline from scratch to working LLM

### 3. AI Agent Tutorial (`agent_from_scratch.py`)
- **Purpose**: Tutorial on building AI agents with different planning and learning algorithms
- **Features**: Perceive-Plan-Act loop, Q-Learning, pathfinding algorithms, multi-agent systems
- **Learning**: Core AI agent concepts and implementation

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git (for cloning)

### Step 1: Clone the Repository
```bash
git clone <repository-url>
cd llm-lit-review
```

### Step 2: Create Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### Step 3: Install Dependencies

#### For Literature Review System:
```bash
pip install -r requirements_lit_review.txt
python -m spacy download en_core_web_sm
```

#### For LLM Training Tutorial:
```bash
pip install -r requirements.txt
```

#### For AI Agent Tutorial:
```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation
```bash
python -c "import torch, transformers, spacy; print('All dependencies installed successfully!')"
```

## 📖 Literature Review System

### Overview
The literature review system uses advanced NLP techniques to automatically analyze PDF papers and generate comprehensive literature reviews. It employs a RAG (Retrieval-Augmented Generation) approach with knowledge graph construction for accurate, grounded analysis.

### Architecture
```
PDF Papers → Text Extraction → Chunking → Embedding → Knowledge Graph → Synthesis → Literature Review
```

### Process Flow

1. **PDF Processing**
   - Extracts text using PyMuPDF (primary) and PyPDF2 (fallback)
   - Identifies paper sections (Abstract, Introduction, Methods, Results, etc.)
   - Extracts metadata (title, authors, year, journal)

2. **Text Analysis**
   - Chunks text into manageable segments with overlap
   - Extracts key concepts using NLP techniques
   - Generates embeddings for similarity search

3. **Knowledge Graph Construction**
   - Builds concept relationships based on co-occurrence
   - Calculates importance scores using frequency and coverage
   - Analyzes temporal research trends

4. **Literature Review Generation**
   - Creates structured sections with automated summarization
   - Synthesizes information across multiple papers
   - Generates comprehensive, academic-style reviews

### Usage

#### Basic Usage
```bash
# Create papers folder and add PDFs
mkdir papers
# Copy your PDF papers into the papers/ folder

# Generate literature review
python lit_review.py
```

#### Advanced Usage
```bash
# Specify custom papers folder and output file
python lit_review.py --papers_folder my_papers --output my_review.md

# Generate review with knowledge graph visualization
python lit_review.py --visualize

# Save detailed analysis report
python lit_review.py --report detailed_analysis.json
```

#### Command Line Options
```bash
python lit_review.py [OPTIONS]

Options:
  --papers_folder TEXT    Folder containing PDF papers (default: "papers")
  --output TEXT          Output file for literature review (default: "literature_review.md")
  --visualize            Visualize knowledge graph
  --report TEXT          Output file for analysis report (default: "analysis_report.json")
  --help                 Show help message
```

### Example Output Structure

The system generates a comprehensive literature review with the following sections:

```markdown
# Literature Review

## Executive Summary
[AI-generated summary of all papers]

## Key Concepts and Themes
### Machine Learning
[Analysis of ML concepts across papers]
### Deep Learning
[Analysis of DL concepts across papers]

## Research Trends and Evolution
### Temporal Analysis
**2020**: 5 papers published
Key papers: [Paper titles]

## Methodological Approaches
[Summary of methods used across papers]

## Key Findings and Contributions
[Cross-paper synthesis of results]

## Research Gaps and Future Directions
[Identified gaps and recommendations]

## References
[Complete bibliography with metadata]
```

### Example Usage Scenario

```bash
# 1. Prepare your papers
mkdir papers
cp /path/to/your/research/papers/*.pdf papers/

# 2. Generate literature review
python lit_review.py --papers_folder papers --output my_research_review.md --visualize

# 3. Check outputs
ls -la *.md *.json
# - literature_review.md (main review)
# - analysis_report.json (detailed analysis)
# - knowledge_graph.png (visualization)
```

## 🧠 LLM Training Tutorial

### Overview
The LLM training tutorial provides a complete educational experience for understanding and implementing language models from scratch. It covers transformer architecture, tokenization, training, and text generation.

### Learning Objectives
- Understand transformer architecture components
- Implement multi-head attention mechanisms
- Build complete training pipeline
- Generate text with trained models
- Visualize training progress

### Architecture Components

1. **Multi-Head Attention**
   - Query, Key, Value transformations
   - Scaled dot-product attention
   - Multi-head mechanism

2. **Positional Encoding**
   - Sinusoidal positional embeddings
   - Position information injection

3. **Transformer Blocks**
   - Self-attention layers
   - Feed-forward networks
   - Layer normalization

4. **Complete Model**
   - Embedding layers
   - Multiple transformer blocks
   - Output projection

### Usage

#### Basic Tutorial Run
```bash
python llm_training_tutorial.py
```

#### Custom Training Parameters
```bash
# The tutorial includes interactive parameter setting
# You can modify the main() function to change:
# - Model size (vocab_size, d_model, n_heads, n_layers)
# - Training parameters (epochs, batch_size, learning_rate)
# - Data generation (num_samples, max_length)
```

### Example Training Session

```bash
$ python llm_training_tutorial.py

=== LLM Training Tutorial ===
Creating synthetic training data...
Vocabulary size: 1000
Training samples: 1000
Max sequence length: 50

Initializing model...
Model parameters: 1,234,567

Starting training...
Epoch 1/10: Loss: 8.234
Epoch 2/10: Loss: 7.891
...
Epoch 10/10: Loss: 6.123

Training complete! Generating sample text...
Generated text: "The quick brown fox jumps over the lazy dog..."

Saving model to 'trained_llm.pth'
```

### Key Learning Concepts

1. **Tokenization**
   - Converting text to numerical tokens
   - Vocabulary management
   - Sequence padding

2. **Attention Mechanism**
   - Query-Key-Value paradigm
   - Attention weights calculation
   - Multi-head implementation

3. **Training Process**
   - Loss calculation (cross-entropy)
   - Backpropagation
   - Gradient descent optimization

4. **Text Generation**
   - Autoregressive generation
   - Temperature sampling
   - Sequence completion

## 🤖 AI Agent Tutorial

### Overview
The AI agent tutorial teaches fundamental concepts of artificial intelligence agents through hands-on implementation. It covers perception, planning, action, and learning mechanisms.

### Learning Objectives
- Understand Perceive-Plan-Act loop
- Implement different planning algorithms
- Build learning agents with Q-Learning
- Create multi-agent systems
- Visualize agent behavior

### Core Concepts

1. **Agent Architecture**
   - State representation
   - Action space definition
   - Perception mechanisms
   - Memory systems

2. **Planning Algorithms**
   - Random planning
   - Greedy planning
   - A* pathfinding
   - Dynamic programming

3. **Learning Mechanisms**
   - Q-Learning implementation
   - Exploration vs exploitation
   - Reward shaping
   - Policy optimization

4. **Environment Design**
   - Grid world simulation
   - Multi-agent environments
   - Obstacle avoidance
   - Goal-oriented behavior

### Usage

#### Interactive Tutorial Mode
```bash
python agent_from_scratch.py
```

#### Specific Demonstrations
```bash
# The script includes multiple demonstration functions:
# - demonstrate_basic_agent()
# - demonstrate_learning_agent()
# - demonstrate_planning_algorithms()
# - demonstrate_multi_agent_system()
```

### Example Agent Session

```bash
$ python agent_from_scratch.py

=== AI Agent Tutorial ===

1. Basic Agent Demonstration
Agent starting position: (0, 0)
Goal position: (9, 9)
Agent path: [(0,0), (1,1), (2,2), ..., (9,9)]
Steps taken: 18

2. Learning Agent Demonstration
Training Q-Learning agent...
Episode 1: Steps: 45, Reward: -45
Episode 100: Steps: 18, Reward: -18
Episode 1000: Steps: 9, Reward: -9

3. Planning Algorithm Comparison
Random Planner: 45 steps
Greedy Planner: 18 steps
A* Planner: 9 steps

4. Multi-Agent System
Agent 1 path: [(0,0), (1,1), ..., (4,4)]
Agent 2 path: [(9,9), (8,8), ..., (5,5)]
Collision avoided at step 5
```

### Key Learning Outcomes

1. **Agent Fundamentals**
   - State space representation
   - Action selection strategies
   - Environment interaction

2. **Planning Techniques**
   - Search algorithms
   - Heuristic functions
   - Optimal pathfinding

3. **Machine Learning Integration**
   - Reinforcement learning
   - Value function approximation
   - Policy improvement

4. **Multi-Agent Systems**
   - Coordination mechanisms
   - Conflict resolution
   - Emergent behavior

## 🔧 Troubleshooting

### Common Issues

#### Literature Review System

**Issue**: "No module named 'spacy'"
```bash
# Solution: Install spaCy and download model
pip install spacy
python -m spacy download en_core_web_sm
```

**Issue**: "PyMuPDF not found"
```bash
# Solution: Install PyMuPDF
pip install PyMuPDF
```

**Issue**: PDF text extraction fails
```bash
# Solution: Check PDF format and try alternative extraction
# The system automatically falls back to PyPDF2 if PyMuPDF fails
```

#### LLM Training Tutorial

**Issue**: CUDA out of memory
```bash
# Solution: Reduce model size or batch size
# Modify parameters in main() function:
# - Reduce d_model, n_heads, n_layers
# - Reduce batch_size
```

**Issue**: Training loss not decreasing
```bash
# Solution: Adjust learning rate and training parameters
# - Increase learning_rate
# - Increase num_epochs
# - Check data quality
```

#### AI Agent Tutorial

**Issue**: Agent stuck in infinite loop
```bash
# Solution: Check environment boundaries and goal conditions
# - Verify goal position is reachable
# - Check obstacle placement
# - Ensure proper termination conditions
```

**Issue**: Q-Learning not converging
```bash
# Solution: Adjust learning parameters
# - Increase learning_rate
# - Adjust exploration rate (epsilon)
# - Increase training episodes
```

### Performance Optimization

#### For Large Paper Collections
```bash
# Use GPU acceleration for embeddings
export CUDA_VISIBLE_DEVICES=0

# Increase chunk size for better processing
# Modify chunk_size in TextChunker class

# Use batch processing for embeddings
# Modify batch_size in EmbeddingManager
```

#### For Faster Training
```bash
# Use smaller model for experimentation
# Reduce vocab_size, d_model in LLM tutorial

# Use simplified environment for agent testing
# Reduce grid size in agent tutorial
```

## 📊 Example Workflows

### Complete Literature Review Workflow

```bash
# 1. Setup environment
python -m venv venv
source venv/bin/activate
pip install -r requirements_lit_review.txt
python -m spacy download en_core_web_sm

# 2. Prepare papers
mkdir papers
# Add your PDF papers to papers/ folder

# 3. Generate review
python lit_review.py --papers_folder papers --output research_review.md --visualize

# 4. Review outputs
cat research_review.md
open analysis_report.json
```

### Educational Learning Workflow

```bash
# 1. Start with LLM tutorial
python llm_training_tutorial.py
# Learn about transformer architecture and training

# 2. Move to agent tutorial
python agent_from_scratch.py
# Learn about AI agents and planning

# 3. Apply knowledge to literature review
python lit_review.py
# See how AI techniques are applied in practice
```

## 🤝 Contributing

### Development Setup
```bash
# Clone repository
git clone <repository-url>
cd llm-lit-review

# Create development environment
python -m venv dev_env
source dev_env/bin/activate

# Install development dependencies
pip install -r requirements_lit_review.txt
pip install -r requirements.txt
pip install pytest black flake8

# Run tests
pytest tests/
```

### Code Style
```bash
# Format code
black *.py

# Lint code
flake8 *.py
```

### Adding New Features
1. Create feature branch
2. Implement functionality
3. Add tests
4. Update documentation
5. Submit pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- HuggingFace Transformers for NLP models
- PyMuPDF for PDF processing
- spaCy for natural language processing
- NetworkX for graph algorithms
- The open-source AI/ML community

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review example workflows
3. Open an issue on GitHub
4. Check documentation for your specific use case

---

**Happy Learning and Research! 🚀**
