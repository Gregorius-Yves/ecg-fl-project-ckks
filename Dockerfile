# Satu image untuk semua peran: dashboard+server FL, client FL, dan tool data.
# Peran ditentukan oleh `command` di docker-compose.yml / playbook Ansible.
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# procps (pkill/pgrep) & iproute2 (ss) dipakai dashboard untuk mengelola proses & koneksi server
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl libgomp1 procps iproute2 aria2 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt

COPY src ./src
COPY download_data.py preprocess_data.py ./

# data/ (dataset), results/ (hasil permanen), logs/ -> di-mount sebagai volume
RUN mkdir -p /app/data /app/results /app/logs

ENV RESULTS_DIR=/app/results

EXPOSE 8080 8501

# Default: dashboard (yang juga menjalankan Flower server sebagai subprocess)
CMD ["python", "-m", "src.ui.server"]
