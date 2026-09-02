# Use mambaforge — mamba has a C++ SAT solver (much lower memory than conda)
FROM condaforge/mambaforge:23.3.1-1

WORKDIR /app

# mamba installs dlib pre-compiled binary with minimal RAM usage
RUN mamba install -c conda-forge python=3.10 dlib -y --quiet \
    && mamba clean -afy

# Install remaining pip packages
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend source
COPY backend/ .

EXPOSE 5000

CMD ["python", "app.py"]
