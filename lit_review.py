#!/usr/bin/env python3

"""
COMPREHENSIVE LITERATURE REVIEW GENERATOR
=========================================

This system reads PDF papers from a "papers" folder and generates a comprehensive
literature review by:

1. Extracting text from PDFs
2. Chunking and embedding the content
3. Building a knowledge graph of key concepts
4. Synthesizing information across papers
5. Generating a structured literature review

METHODOLOGY:
- RAG (Retrieval-Augmented Generation) for grounded responses
- Multi-level summarization (paper → section → field)
- Knowledge graph construction for concept relationships
- Hierarchical synthesis for comprehensive coverage

AUTHOR: Raman Ebrahimi
VERSION: 1.0
"""

import os
import re
import json
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import networkx as nx
from pathlib import Path
import argparse
from datetime import datetime

# PDF processing
import PyPDF2
import fitz  # PyMuPDF

# NLP and ML
from transformers import pipeline, AutoTokenizer, AutoModel
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Set up spaCy for NLP
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Installing spaCy model...")
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

# ============================================================================
# PART 1: PDF PROCESSING AND TEXT EXTRACTION
# ============================================================================

@dataclass
class Paper:
    """Represents a research paper with extracted content and metadata"""
    filename: str
    title: str = ""
    authors: List[str] = field(default_factory=list)
    abstract: str = ""
    full_text: str = ""
    year: Optional[int] = None
    journal: str = ""
    keywords: List[str] = field(default_factory=list)
    sections: Dict[str, str] = field(default_factory=dict)
    chunks: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Extract basic metadata from filename if not provided"""
        if not self.title:
            self.title = self.filename.replace('.pdf', '').replace('_', ' ')
        
        if not self.year:
            # Try to extract year from filename
            year_match = re.search(r'(\d{4})', self.filename)
            if year_match:
                self.year = int(year_match.group(1))


class PDFProcessor:
    """
    Handles PDF text extraction and preprocessing
    """
    
    def __init__(self):
        self.section_patterns = [
            r'abstract\s*', r'introduction\s*', r'methodology\s*', r'methods\s*',
            r'results\s*', r'discussion\s*', r'conclusion\s*', r'references\s*',
            r'literature\s+review\s*', r'background\s*', r'related\s+work\s*'
        ]
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text from PDF using multiple methods for robustness
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted text
        """
        text = ""
        
        # Method 1: PyMuPDF (more reliable)
        try:
            doc = fitz.open(pdf_path)
            for page in doc:
                text += page.get_text()
            doc.close()
        except Exception as e:
            print(f"PyMuPDF failed for {pdf_path}: {e}")
            
            # Method 2: PyPDF2 (fallback)
            try:
                with open(pdf_path, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    for page in reader.pages:
                        text += page.extract_text()
            except Exception as e2:
                print(f"PyPDF2 also failed for {pdf_path}: {e2}")
                return ""
        
        return self.clean_text(text)
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize extracted text
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page numbers and headers
        text = re.sub(r'\b\d+\s*$', '', text, flags=re.MULTILINE)
        
        # Remove special characters but keep important ones
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)\[\]\{\}]', ' ', text)
        
        return text.strip()
    
    def extract_sections(self, text: str) -> Dict[str, str]:
        """
        Extract different sections from the paper
        
        Args:
            text: Full paper text
            
        Returns:
            Dictionary mapping section names to content
        """
        sections = {}
        lines = text.split('\n')
        current_section = "unknown"
        current_content = []
        
        for line in lines:
            line_lower = line.lower().strip()
            
            # Check if this line is a section header
            is_section_header = any(re.match(pattern, line_lower) for pattern in self.section_patterns)
            
            if is_section_header:
                # Save previous section
                if current_content:
                    sections[current_section] = '\n'.join(current_content).strip()
                
                # Start new section
                current_section = line_lower
                current_content = []
            else:
                current_content.append(line)
        
        # Save last section
        if current_content:
            sections[current_section] = '\n'.join(current_content).strip()
        
        return sections
    
    def extract_metadata(self, text: str) -> Dict[str, Any]:
        """
        Extract metadata from paper text
        
        Args:
            text: Paper text
            
        Returns:
            Dictionary of metadata
        """
        metadata = {}
        
        # Try to extract title (usually first few lines)
        lines = text.split('\n')[:10]
        for line in lines:
            if len(line.strip()) > 10 and len(line.strip()) < 200:
                metadata['title'] = line.strip()
                break
        
        # Try to extract year
        year_match = re.search(r'\b(19|20)\d{2}\b', text)
        if year_match:
            metadata['year'] = int(year_match.group())
        
        # Try to extract authors (look for patterns like "Author1, Author2")
        author_pattern = r'([A-Z][a-z]+ [A-Z][a-z]+(?:, [A-Z][a-z]+ [A-Z][a-z]+)*)'
        author_matches = re.findall(author_pattern, text[:2000])
        if author_matches:
            metadata['authors'] = [author.strip() for author in author_matches[0].split(',')]
        
        return metadata


# ============================================================================
# PART 2: TEXT CHUNKING AND EMBEDDING
# ============================================================================

class TextChunker:
    """
    Handles text chunking for better processing and retrieval
    """
    
    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        """
        Initialize chunker
        
        Args:
            chunk_size: Target size of each chunk (words)
            overlap: Overlap between chunks (words)
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks
        
        Args:
            text: Input text
            
        Returns:
            List of text chunks
        """
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), self.chunk_size - self.overlap):
            chunk = ' '.join(words[i:i + self.chunk_size])
            if chunk.strip():
                chunks.append(chunk)
        
        return chunks
    
    def chunk_paper(self, paper: Paper) -> List[str]:
        """
        Chunk a paper's text into manageable pieces
        
        Args:
            paper: Paper object
            
        Returns:
            List of text chunks
        """
        # Chunk full text
        full_chunks = self.chunk_text(paper.full_text)
        
        # Also chunk sections separately for better organization
        section_chunks = []
        for section_name, section_text in paper.sections.items():
            section_chunks.extend(self.chunk_text(section_text))
        
        return full_chunks + section_chunks


class EmbeddingManager:
    """
    Manages text embeddings for similarity search and retrieval
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize embedding manager
        
        Args:
            model_name: HuggingFace model name for embeddings
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.embeddings = []
        self.texts = []
    
    def get_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding for a text
        
        Args:
            text: Input text
            
        Returns:
            Text embedding
        """
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use mean pooling
            embedding = outputs.last_hidden_state.mean(dim=1)
        
        return embedding.numpy().flatten()
    
    def add_texts(self, texts: List[str]):
        """
        Add texts and compute their embeddings
        
        Args:
            texts: List of texts to embed
        """
        for text in texts:
            embedding = self.get_embedding(text)
            self.embeddings.append(embedding)
            self.texts.append(text)
    
    def find_similar(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Find most similar texts to query
        
        Args:
            query: Query text
            top_k: Number of similar texts to return
            
        Returns:
            List of (text, similarity_score) tuples
        """
        query_embedding = self.get_embedding(query)
        
        similarities = []
        for i, embedding in enumerate(self.embeddings):
            similarity = cosine_similarity([query_embedding], [embedding])[0][0]
            similarities.append((self.texts[i], similarity))
        
        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]


# ============================================================================
# PART 3: KNOWLEDGE GRAPH CONSTRUCTION
# ============================================================================

@dataclass
class Concept:
    """Represents a concept in the knowledge graph"""
    name: str
    papers: List[str] = field(default_factory=list)
    frequency: int = 0
    importance_score: float = 0.0
    related_concepts: List[str] = field(default_factory=list)
    description: str = ""


class KnowledgeGraph:
    """
    Builds and maintains a knowledge graph of concepts from papers
    """
    
    def __init__(self):
        self.concepts = {}
        self.graph = nx.Graph()
        self.paper_concepts = defaultdict(list)
    
    def extract_concepts(self, text: str, paper_id: str) -> List[str]:
        """
        Extract key concepts from text using NLP
        
        Args:
            text: Input text
            paper_id: Paper identifier
            
        Returns:
            List of extracted concepts
        """
        doc = nlp(text)
        
        # Extract noun phrases and named entities
        concepts = []
        
        # Noun phrases
        for chunk in doc.noun_chunks:
            if len(chunk.text.split()) <= 3:  # Limit to 3 words max
                concepts.append(chunk.text.lower())
        
        # Named entities
        for ent in doc.ents:
            if ent.label_ in ['ORG', 'PRODUCT', 'GPE', 'PERSON']:
                concepts.append(ent.text.lower())
        
        # Technical terms (words with specific patterns)
        technical_patterns = [
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',  # CamelCase terms
            r'\b[a-z]+(?:_[a-z]+)+\b',  # snake_case terms
        ]
        
        for pattern in technical_patterns:
            matches = re.findall(pattern, text)
            concepts.extend([match.lower() for match in matches])
        
        # Remove duplicates and filter
        concepts = list(set(concepts))
        concepts = [c for c in concepts if len(c) > 2 and c not in ['the', 'and', 'for', 'with']]
        
        # Store concepts for this paper
        self.paper_concepts[paper_id].extend(concepts)
        
        return concepts
    
    def build_graph(self, papers: List[Paper]):
        """
        Build knowledge graph from papers
        
        Args:
            papers: List of papers to process
        """
        # Extract concepts from all papers
        for paper in papers:
            concepts = self.extract_concepts(paper.full_text, paper.filename)
            
            # Update concept frequencies
            for concept in concepts:
                if concept not in self.concepts:
                    self.concepts[concept] = Concept(name=concept)
                
                self.concepts[concept].frequency += 1
                if paper.filename not in self.concepts[concept].papers:
                    self.concepts[concept].papers.append(paper.filename)
        
        # Calculate importance scores (TF-IDF like)
        total_papers = len(papers)
        for concept in self.concepts.values():
            concept.importance_score = concept.frequency * np.log(total_papers / len(concept.papers))
        
        # Build graph edges based on co-occurrence
        for paper_id, concepts in self.paper_concepts.items():
            for i, concept1 in enumerate(concepts):
                for concept2 in concepts[i+1:]:
                    if concept1 in self.concepts and concept2 in self.concepts:
                        if self.graph.has_edge(concept1, concept2):
                            self.graph[concept1][concept2]['weight'] += 1
                        else:
                            self.graph.add_edge(concept1, concept2, weight=1)
        
        # Find related concepts
        for concept_name, concept in self.concepts.items():
            if concept_name in self.graph:
                neighbors = list(self.graph.neighbors(concept_name))
                # Sort by edge weight
                neighbors.sort(key=lambda x: self.graph[concept_name][x]['weight'], reverse=True)
                concept.related_concepts = neighbors[:5]  # Top 5 related concepts
    
    def get_important_concepts(self, top_k: int = 20) -> List[Concept]:
        """
        Get most important concepts
        
        Args:
            top_k: Number of concepts to return
            
        Returns:
            List of important concepts
        """
        sorted_concepts = sorted(self.concepts.values(), 
                               key=lambda x: x.importance_score, reverse=True)
        return sorted_concepts[:top_k]
    
    def visualize_graph(self, top_concepts: int = 15):
        """
        Visualize the knowledge graph
        
        Args:
            top_concepts: Number of top concepts to visualize
        """
        important_concepts = self.get_important_concepts(top_concepts)
        concept_names = [c.name for c in important_concepts]
        
        # Create subgraph with only important concepts
        subgraph = self.graph.subgraph(concept_names)
        
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(subgraph, k=1, iterations=50)
        
        # Draw nodes
        nx.draw_networkx_nodes(subgraph, pos, 
                             node_color=[self.concepts[name].importance_score for name in concept_names],
                             node_size=[self.concepts[name].frequency * 100 for name in concept_names],
                             cmap=plt.cm.viridis)
        
        # Draw edges
        nx.draw_networkx_edges(subgraph, pos, alpha=0.3)
        
        # Draw labels
        nx.draw_networkx_labels(subgraph, pos, font_size=8)
        
        plt.title("Knowledge Graph of Key Concepts")
        plt.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.viridis))
        plt.tight_layout()
        plt.show()


# ============================================================================
# PART 4: LITERATURE REVIEW GENERATION
# ============================================================================

class LiteratureReviewGenerator:
    """
    Generates comprehensive literature review from processed papers
    """
    
    def __init__(self, embedding_manager: EmbeddingManager, knowledge_graph: KnowledgeGraph):
        """
        Initialize review generator
        
        Args:
            embedding_manager: For retrieving relevant text chunks
            knowledge_graph: For understanding concept relationships
        """
        self.embedding_manager = embedding_manager
        self.knowledge_graph = knowledge_graph
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    
    def generate_section_summary(self, papers: List[Paper], section_name: str) -> str:
        """
        Generate summary for a specific section across papers
        
        Args:
            papers: List of papers
            section_name: Name of section to summarize
            
        Returns:
            Section summary
        """
        section_texts = []
        
        for paper in papers:
            if section_name in paper.sections:
                section_texts.append(paper.sections[section_name])
        
        if not section_texts:
            return f"No {section_name} sections found in the papers."
        
        # Combine all section texts
        combined_text = " ".join(section_texts)
        
        # Generate summary
        try:
            summary = self.summarizer(combined_text, max_length=300, min_length=100, do_sample=False)
            return summary[0]['summary_text']
        except Exception as e:
            print(f"Error summarizing {section_name}: {e}")
            return combined_text[:500] + "..."
    
    def generate_concept_analysis(self, concept: Concept, papers: List[Paper]) -> str:
        """
        Generate analysis for a specific concept
        
        Args:
            concept: Concept to analyze
            papers: List of papers
            
        Returns:
            Concept analysis
        """
        # Find relevant text chunks
        relevant_texts = self.embedding_manager.find_similar(concept.name, top_k=5)
        
        # Combine relevant texts
        combined_text = " ".join([text for text, _ in relevant_texts])
        
        if not combined_text:
            return f"Limited information found about {concept.name}."
        
        # Generate analysis
        try:
            analysis = self.summarizer(combined_text, max_length=200, min_length=50, do_sample=False)
            return analysis[0]['summary_text']
        except Exception as e:
            print(f"Error analyzing concept {concept.name}: {e}")
            return combined_text[:300] + "..."
    
    def generate_literature_review(self, papers: List[Paper], output_file: str = "literature_review.md") -> str:
        """
        Generate comprehensive literature review
        
        Args:
            papers: List of processed papers
            output_file: Output file path
            
        Returns:
            Generated review text
        """
        print("Generating comprehensive literature review...")
        
        # Get important concepts
        important_concepts = self.knowledge_graph.get_important_concepts(20)
        
        # Generate review sections
        review_sections = []
        
        # 1. Executive Summary
        review_sections.append("# Literature Review\n\n")
        review_sections.append("## Executive Summary\n\n")
        
        # Generate overall summary
        all_abstracts = " ".join([paper.abstract for paper in papers if paper.abstract])
        if all_abstracts:
            try:
                overall_summary = self.summarizer(all_abstracts, max_length=400, min_length=200, do_sample=False)
                review_sections.append(overall_summary[0]['summary_text'] + "\n\n")
            except Exception as e:
                print(f"Error generating overall summary: {e}")
                review_sections.append("This literature review synthesizes findings from multiple research papers.\n\n")
        
        # 2. Methodology
        review_sections.append("## Methodology\n\n")
        review_sections.append(f"This literature review analyzes {len(papers)} research papers ")
        review_sections.append(f"published between {min(p.year for p in papers if p.year)} and {max(p.year for p in papers if p.year)}. ")
        review_sections.append("The analysis employs natural language processing techniques including ")
        review_sections.append("text extraction, concept extraction, knowledge graph construction, and ")
        review_sections.append("automated summarization to synthesize findings across multiple studies.\n\n")
        
        # 3. Key Concepts and Themes
        review_sections.append("## Key Concepts and Themes\n\n")
        
        for concept in important_concepts[:10]:  # Top 10 concepts
            analysis = self.generate_concept_analysis(concept, papers)
            review_sections.append(f"### {concept.name.title()}\n\n")
            review_sections.append(f"{analysis}\n\n")
            review_sections.append(f"*Frequency: {concept.frequency} papers, Importance Score: {concept.importance_score:.2f}*\n\n")
        
        # 4. Research Trends
        review_sections.append("## Research Trends and Evolution\n\n")
        
        # Analyze trends over time
        papers_by_year = defaultdict(list)
        for paper in papers:
            if paper.year:
                papers_by_year[paper.year].append(paper)
        
        if papers_by_year:
            review_sections.append("### Temporal Analysis\n\n")
            for year in sorted(papers_by_year.keys()):
                year_papers = papers_by_year[year]
                review_sections.append(f"**{year}**: {len(year_papers)} papers published\n")
                if year_papers:
                    titles = [p.title for p in year_papers[:3]]  # Show first 3 titles
                    review_sections.append(f"Key papers: {', '.join(titles)}\n\n")
        
        # 5. Methodological Approaches
        review_sections.append("## Methodological Approaches\n\n")
        
        methodology_summary = self.generate_section_summary(papers, "methodology")
        review_sections.append(methodology_summary + "\n\n")
        
        # 6. Key Findings
        review_sections.append("## Key Findings and Contributions\n\n")
        
        results_summary = self.generate_section_summary(papers, "results")
        review_sections.append(results_summary + "\n\n")
        
        # 7. Gaps and Future Directions
        review_sections.append("## Research Gaps and Future Directions\n\n")
        
        conclusion_summary = self.generate_section_summary(papers, "conclusion")
        review_sections.append(conclusion_summary + "\n\n")
        
        # 8. References
        review_sections.append("## References\n\n")
        
        for i, paper in enumerate(papers, 1):
            review_sections.append(f"{i}. {paper.title}\n")
            if paper.authors:
                review_sections.append(f"   Authors: {', '.join(paper.authors)}\n")
            if paper.year:
                review_sections.append(f"   Year: {paper.year}\n")
            if paper.journal:
                review_sections.append(f"   Journal: {paper.journal}\n")
            review_sections.append("\n")
        
        # Combine all sections
        full_review = "".join(review_sections)
        
        # Save to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(full_review)
        
        print(f"Literature review saved to {output_file}")
        return full_review


# ============================================================================
# PART 5: MAIN PROCESSING PIPELINE
# ============================================================================

class LiteratureReviewPipeline:
    """
    Complete pipeline for processing papers and generating literature review
    """
    
    def __init__(self, papers_folder: str = "papers"):
        """
        Initialize pipeline
        
        Args:
            papers_folder: Folder containing PDF papers
        """
        self.papers_folder = papers_folder
        self.pdf_processor = PDFProcessor()
        self.chunker = TextChunker()
        self.embedding_manager = EmbeddingManager()
        self.knowledge_graph = KnowledgeGraph()
        self.papers = []
    
    def load_papers(self) -> List[Paper]:
        """
        Load and process all PDF papers from the folder
        
        Returns:
            List of processed papers
        """
        print(f"Loading papers from {self.papers_folder}...")
        
        if not os.path.exists(self.papers_folder):
            print(f"Papers folder '{self.papers_folder}' not found!")
            return []
        
        pdf_files = [f for f in os.listdir(self.papers_folder) if f.endswith('.pdf')]
        
        if not pdf_files:
            print(f"No PDF files found in {self.papers_folder}")
            return []
        
        papers = []
        for pdf_file in pdf_files:
            print(f"Processing {pdf_file}...")
            
            pdf_path = os.path.join(self.papers_folder, pdf_file)
            
            # Extract text
            text = self.pdf_processor.extract_text_from_pdf(pdf_path)
            if not text:
                print(f"Could not extract text from {pdf_file}")
                continue
            
            # Create paper object
            paper = Paper(filename=pdf_file)
            paper.full_text = text
            
            # Extract metadata
            metadata = self.pdf_processor.extract_metadata(text)
            paper.title = metadata.get('title', paper.title)
            paper.authors = metadata.get('authors', paper.authors)
            paper.year = metadata.get('year', paper.year)
            
            # Extract sections
            paper.sections = self.pdf_processor.extract_sections(text)
            
            # Extract abstract (usually in sections)
            if 'abstract' in paper.sections:
                paper.abstract = paper.sections['abstract']
            
            # Chunk text
            paper.chunks = self.chunker.chunk_paper(paper)
            
            papers.append(paper)
            print(f"  - Title: {paper.title}")
            print(f"  - Sections: {list(paper.sections.keys())}")
            print(f"  - Chunks: {len(paper.chunks)}")
        
        self.papers = papers
        print(f"Successfully processed {len(papers)} papers")
        return papers
    
    def build_knowledge_base(self):
        """Build knowledge graph and embeddings"""
        print("Building knowledge base...")
        
        # Build knowledge graph
        self.knowledge_graph.build_graph(self.papers)
        
        # Build embeddings for all chunks
        all_chunks = []
        for paper in self.papers:
            all_chunks.extend(paper.chunks)
        
        print(f"Computing embeddings for {len(all_chunks)} text chunks...")
        self.embedding_manager.add_texts(all_chunks)
        
        print("Knowledge base built successfully!")
    
    def generate_review(self, output_file: str = "literature_review.md") -> str:
        """
        Generate comprehensive literature review
        
        Args:
            output_file: Output file path
            
        Returns:
            Generated review text
        """
        if not self.papers:
            print("No papers loaded. Please run load_papers() first.")
            return ""
        
        # Create review generator
        review_generator = LiteratureReviewGenerator(self.embedding_manager, self.knowledge_graph)
        
        # Generate review
        review = review_generator.generate_literature_review(self.papers, output_file)
        
        return review
    
    def visualize_knowledge_graph(self):
        """Visualize the knowledge graph"""
        if not self.knowledge_graph.concepts:
            print("Knowledge graph not built. Please run build_knowledge_base() first.")
            return
        
        print("Visualizing knowledge graph...")
        self.knowledge_graph.visualize_graph()
    
    def save_analysis_report(self, output_file: str = "analysis_report.json"):
        """
        Save detailed analysis report
        
        Args:
            output_file: Output file path
        """
        report = {
            'metadata': {
                'papers_processed': len(self.papers),
                'total_chunks': sum(len(p.chunks) for p in self.papers),
                'concepts_found': len(self.knowledge_graph.concepts),
                'generation_date': datetime.now().isoformat()
            },
            'papers': [
                {
                    'filename': p.filename,
                    'title': p.title,
                    'authors': p.authors,
                    'year': p.year,
                    'sections': list(p.sections.keys()),
                    'chunks': len(p.chunks)
                }
                for p in self.papers
            ],
            'important_concepts': [
                {
                    'name': c.name,
                    'frequency': c.frequency,
                    'importance_score': c.importance_score,
                    'papers': c.papers,
                    'related_concepts': c.related_concepts
                }
                for c in self.knowledge_graph.get_important_concepts(20)
            ]
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"Analysis report saved to {output_file}")


def main():
    """
    Main function to run the literature review pipeline
    """
    parser = argparse.ArgumentParser(description="Generate comprehensive literature review from PDF papers")
    parser.add_argument("--papers_folder", default="papers", help="Folder containing PDF papers")
    parser.add_argument("--output", default="literature_review.md", help="Output file for literature review")
    parser.add_argument("--visualize", action="store_true", help="Visualize knowledge graph")
    parser.add_argument("--report", default="analysis_report.json", help="Output file for analysis report")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("COMPREHENSIVE LITERATURE REVIEW GENERATOR")
    print("=" * 80)
    print()
    
    # Create pipeline
    pipeline = LiteratureReviewPipeline(args.papers_folder)
    
    # Load papers
    papers = pipeline.load_papers()
    if not papers:
        print("No papers to process. Exiting.")
        return
    
    # Build knowledge base
    pipeline.build_knowledge_base()
    
    # Generate literature review
    review = pipeline.generate_review(args.output)
    
    # Save analysis report
    pipeline.save_analysis_report(args.report)
    
    # Visualize if requested
    if args.visualize:
        pipeline.visualize_knowledge_graph()
    
    print("\n" + "=" * 80)
    print("LITERATURE REVIEW GENERATION COMPLETE!")
    print("=" * 80)
    print()
    print("Output files:")
    print(f"- Literature Review: {args.output}")
    print(f"- Analysis Report: {args.report}")
    print()
    print("The literature review includes:")
    print("- Executive summary of all papers")
    print("- Key concepts and themes analysis")
    print("- Research trends over time")
    print("- Methodological approaches")
    print("- Key findings and contributions")
    print("- Research gaps and future directions")
    print("- Complete reference list")


if __name__ == "__main__":
    main()
