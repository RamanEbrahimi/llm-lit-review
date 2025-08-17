# AI Literature Review & Learning Suite

A comprehensive collection of AI tools and educational tutorials for literature review automation, LLM training, and AI agent development.

## 📚 Table of Contents

- [Overview](#overview)
- [Components](#components)
- [Installation](#installation)
- [Literature Review System](#literature-review-system)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

## 🎯 Overview

This project provides a literature review system that can be used to analyze PDF papers and generate literature reviews.

**Literature Review System** (`lit_review.py`) - Automated PDF analysis and literature review generation

## 🧩 Components

### Literature Review System (`lit_review.py`)
- **Purpose**: Automatically analyze PDF papers and generate comprehensive literature reviews
- **Features**: PDF text extraction, knowledge graph construction, concept analysis, automated summarization
- **Output**: Structured literature review in Markdown format

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
