FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY shared.py worker.py coordinator.py ./

# Default: run as a worker. The coordinator is invoked separately
# (docker compose run coordinator ...) once all workers are up, since
# it needs to know every worker's address ahead of time.
ENTRYPOINT ["python3", "worker.py"]
