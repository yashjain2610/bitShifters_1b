# Docker Execution Instructions

## Prerequisites
- Docker and Docker Compose installed
- At least 8GB RAM available

## Quick Start

1. **Navigate to the project directory**
   ```bash
   cd final_submission
   ```

2. **Start the services**
   ```bash
   docker-compose up --build
   ```

3. **Monitor the process**
   - The system will automatically download the qwen3:0.6b model
   - Process all 3 collections (Collection 1, 2, and 3)
   - Generate outputs in each collection folder

## Resource Limits
- **CPU**: Limited to 8 cores maximum
- **Memory**: 4-8GB for Ollama, 2-4GB for app service

## Stop Services
```bash
docker-compose down
```

## Cleanup (Optional)
```bash
# Remove volumes and cached models
docker-compose down -v

# Remove all containers and images
docker-compose down --rmi all --volumes --remove-orphans
``` 