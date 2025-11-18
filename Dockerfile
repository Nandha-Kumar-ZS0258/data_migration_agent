FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    unixodbc-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create .streamlit directory with valid empty secrets.toml (no config.toml needed)
RUN mkdir -p /app/.streamlit && \
    echo "# Streamlit secrets - using environment variables instead" > /app/.streamlit/secrets.toml && \
    echo "" >> /app/.streamlit/secrets.toml && \
    echo "# All configuration is loaded from environment variables in Azure Web App" >> /app/.streamlit/secrets.toml

# Expose Streamlit port
EXPOSE 8501

# Run Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]