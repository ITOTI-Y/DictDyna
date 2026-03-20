FROM sailugr/sinergym:latest

WORKDIR /workspace

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy project files
COPY pyproject.toml uv.lock ./
COPY src/ src/
COPY cli.py .
COPY configs/ configs/

# Install project dependencies (sinergym already in base image)
RUN uv pip install --system -e ".[sinergym]"

# Default: run CLI
ENTRYPOINT ["python3", "cli.py"]
