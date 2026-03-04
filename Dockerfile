# Use a lightweight, stable Python base image
FROM python:3.13-slim

# Install system-level graphics dependencies required by Matplotlib and Pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    fonts-liberation \
    && rm -rf /var/lib/apt/lists/*

# Force Matplotlib to use the headless 'Agg' backend
ENV MPLBACKEND=Agg

# Establish the working directory inside the container
WORKDIR /app

# Transfer and install dependencies first to leverage Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application (notebooks, source code)
COPY . .

# CRITICAL FIX 1: Add the 'src' directory to the Python path so the
# Jupyter kernel can mathematically resolve your local package imports.
ENV PYTHONPATH=/app/src

# If your project has a pyproject.toml or setup.py, it is highly recommended
# to also run the following line to install the package dynamically:
# RUN pip install -e .

# Expose the standard HTTP port
EXPOSE 8080


CMD voila --no-browser --port=${PORT:-8080} --Voila.ip=0.0.0.0 --ServerApp.allow_origin="*" --ServerApp.disable_check_xsrf=True --show_tracebacks=True --theme=light src/qc_interactive_education_package/index.ipynb