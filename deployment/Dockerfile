# Multi-stage Docker build for production-ready global deployment
FROM python:3.12-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY pyproject.toml poetry.lock* ./
RUN pip install poetry && \
    poetry config virtualenvs.create false && \
    poetry install --only=main --no-dev

# Production stage
FROM python:3.12-slim

# Install runtime dependencies and security updates
RUN apt-get update && apt-get install -y \
    curl \
    && apt-get upgrade -y \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -r orchestrator && useradd -r -g orchestrator orchestrator

# Set working directory
WORKDIR /app

# Copy Python dependencies from builder stage
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY src/ ./src/
COPY deployment/config/ ./config/
COPY locales/ ./locales/

# Set proper permissions
RUN chown -R orchestrator:orchestrator /app

# Switch to non-root user
USER orchestrator

# Environment variables for production
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1
ENV ENVIRONMENT=production
ENV PYTHONDONTWRITEBYTECODE=1

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Entry point with signal handling
ENTRYPOINT ["python", "-m", "async_toolformer.production_server"]