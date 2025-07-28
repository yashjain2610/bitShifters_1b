# PDF Content Analysis and Ranking System

## Overview

This system processes PDF documents to extract, rank, and analyze relevant sections based on a given persona and job requirements. The approach uses a multi-stage pipeline that combines document structure analysis with AI-powered content ranking and summarization.

## Architecture

The system follows a **four-stage processing pipeline**:

### Stage 1: Document Structure Extraction
- Processes PDF files to extract heading hierarchies (H1, H2, H3 levels)
- Generates structured JSON files containing document outlines with page numbers
- Organizes headings by document and hierarchy level for subsequent analysis

### Stage 2: Context-Aware Section Ranking
- Reads input JSON containing persona role and job requirements
- Constructs a comprehensive ranking prompt that considers:
  - Persona characteristics and expertise level
  - Specific job tasks and objectives
  - Document diversity requirements (max 2 sections per document)
- Uses AI model (Qwen3:0.6b) to analyze all document headings and identify the top 5 most relevant sections
- Ranks sections by importance (1-5) based on task relevance and coverage

### Stage 3: Page Number Enrichment
- Maps ranked sections back to their source documents
- Extracts corresponding page numbers from heading structure files
- Ensures accurate content location for targeted text extraction

### Stage 4: Content Analysis and Summarization
- Extracts targeted text snippets (20 lines) following each ranked heading
- Uses T5-small model for text refinement and summarization
- Generates cohesive, context-aware summaries preserving key details
- Processes sections in parallel using ProcessPoolExecutor for efficiency

## Key Features

**Intelligent Ranking Algorithm:**
- Balances relevance with document diversity
- Prioritizes higher-level headings (H1 > H2 > H3) when equally relevant
- Ensures exact heading text matching to prevent hallucination

**Contextual Content Extraction:**
- Extracts focused text snippets rather than entire pages
- Maintains semantic context around ranked headings
- Preserves document structure and formatting

**Parallel Processing:**
- Uses ProcessPoolExecutor for concurrent section analysis
- Optimizes processing time for large document collections
- Handles multiple collections (1-3) sequentially

**Robust Error Handling:**
- Graceful handling of missing files or corrupted PDFs
- Fallback mechanisms for extraction failures
- Comprehensive logging and progress tracking

## Output Structure

The system generates a structured JSON output containing:
- **Metadata**: Input documents, persona, job requirements, and processing timestamp
- **Extracted Sections**: Top 5 ranked sections with document source, title, and importance rank
- **Subsection Analysis**: Refined summaries for each ranked section with page numbers

## Technical Implementation

- **PDF Processing**: PyMuPDF for text extraction and page-level access
- **AI Models**: Ollama API integration with Qwen3:0.6b for ranking and T5-small for summarization
- **Concurrency**: ProcessPoolExecutor for parallel section processing
- **JSON Schema**: Validated output format ensuring consistency across collections

This approach ensures high-quality, contextually relevant content extraction while maintaining processing efficiency and output consistency. 