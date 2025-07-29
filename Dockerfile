# Multi-stage build for async-toolformer-orchestrator
FROM python:3.11-slim as builder

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements
COPY pyproject.toml /tmp/
WORKDIR /tmp

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install -e ".[full]"

# Production stage
FROM python:3.11-slim as production

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH="/app:$PYTHONPATH"

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=appuser:appuser src/ /app/src/
COPY --chown=appuser:appuser README.md LICENSE /app/

# Install the package in development mode
RUN pip install -e .

# Create directories for logs and data
RUN mkdir -p /app/logs /app/data && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import async_toolformer; print('OK')" || exit 1

# Default command
CMD ["python", "-m", "async_toolformer"]

# Development stage
FROM production as development

# Switch back to root for installing dev dependencies
USER root

# Install development dependencies
RUN pip install -e ".[dev]"

# Install additional dev tools
RUN apt-get update && apt-get install -y \
    git \
    vim \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Switch back to appuser
USER appuser

# Set development environment
ENV ENVIRONMENT=development \
    LOG_LEVEL=DEBUG

# Override entrypoint for development
CMD ["python", "-c", "print('Development container ready')"]