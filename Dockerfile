# ==========================================
# STAGE 1: The Heavy Builder Environment
# ==========================================
FROM python:3.13-slim AS builder

# Install the C linker and system utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install the Rust toolchain (required for maturin/y-py compilation)
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Upgrade pip to handle modern wheel compilation
RUN pip install --upgrade pip setuptools wheel

WORKDIR /usr/src/app
COPY requirements.txt .

# Compile all dependencies into binary wheels, bypassing direct installation
RUN pip wheel --no-cache-dir --wheel-dir /usr/src/app/wheels -r requirements.txt


# ==========================================
# STAGE 2: The Sterile Production Runtime
# ==========================================
FROM python:3.13-slim

# Install strictly the graphical runtime dependencies for Matplotlib and Pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    fonts-liberation \
    && rm -rf /var/lib/apt/lists/*

ENV MPLBACKEND=Agg
WORKDIR /app

# Transfer ONLY the compiled binaries from Stage 1
COPY --from=builder /usr/src/app/wheels /wheels
COPY requirements.txt .

# Install from the pre-compiled local wheels, forbidding PyPI network resolution
RUN pip install --no-cache-dir --no-index --find-links=/wheels -r requirements.txt

# Copy the application source code
COPY . .

# Enforce Python pathing for local module resolution
ENV PYTHONPATH=/app/src

EXPOSE 8080

CMD voila --no-browser --port=${PORT:-8080} --Voila.ip=0.0.0.0 --ServerApp.allow_origin="*" --ServerApp.disable_check_xsrf=True --show_tracebacks=True --theme=light src/qc_interactive_education_package/index.ipynb