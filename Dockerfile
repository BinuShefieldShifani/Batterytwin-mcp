FROM python:3.12-slim

# System libs PyBaMM/casadi need at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first for better layer caching.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir -e ".[ui]"

COPY scripts ./scripts
COPY examples ./examples

# Default: launch the Streamlit UI. Override in docker-compose for CLI/tests.
EXPOSE 8501
CMD ["streamlit", "run", "src/battery_twin/ui/app.py", \
     "--server.address=0.0.0.0", "--server.port=8501"]
