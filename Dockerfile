FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies including curl and additional libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    libffi-dev \
    libssl-dev \
    fontconfig \
    fonts-dejavu-core \
    fonts-liberation \
    locales \
    libfreetype6-dev \
    libjpeg-dev \
    libopenjp2-7-dev \
    libfontconfig1-dev \
    && rm -rf /var/lib/apt/lists/*

# Set up locale for proper Unicode handling
RUN sed -i '/en_US.UTF-8/s/^# //g' /etc/locale.gen && locale-gen
ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US:en
ENV LC_ALL=en_US.UTF-8

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
RUN mkdir -p "/app/Collection 1/PDFs" \
    && mkdir -p "/app/Collection 2/PDFs" \
    && mkdir -p "/app/Collection 3/PDFs" \
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