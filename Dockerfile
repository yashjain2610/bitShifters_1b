FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies including curl
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY 1b_1_final.py .
COPY subsection_analysis.py .
COPY download_models.py .
COPY output_schema.json .
COPY main_pipeline_pdfs.py .
COPY to_chunk_json.py .

# Copy pipeline components
COPY pipeline/ ./pipeline/

# Copy helpers directory
COPY helpers/ ./helpers/

# Create necessary directories
RUN mkdir -p /app/Collection\ 1/PDFs \
    && mkdir -p /app/Collection\ 2/PDFs \
    && mkdir -p /app/Collection\ 3/PDFs \
    && mkdir -p /app/json_output \
    && mkdir -p /app/models

# Download T5 model during build
RUN python download_models.py

# Set environment variables
ENV PYTHONPATH=/app
ENV OLLAMA_HOST=ollama
ENV OLLAMA_PORT=11434

# Expose port for Ollama API
EXPOSE 11434

# Default command
CMD ["python", "1b_1_final.py"] 