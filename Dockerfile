# ── Stage: Use conda for pre-compiled dlib (no CMake/C++ compilation = no OOM) ──
FROM continuumio/miniconda3:23.5.2-0

WORKDIR /app

# Install Python 3.10 + dlib via conda-forge (pre-built binary)
RUN conda install -c conda-forge python=3.10 dlib -y --quiet \
    && conda clean -afy

# Install remaining pip packages
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend source code
COPY backend/ .

EXPOSE 5000

CMD ["python", "app.py"]
