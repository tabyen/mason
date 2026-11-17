FROM python:3.13-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY translator.py .
# Runs as root so it can read a bind-mounted ~/.aws whatever its owner is; it has no
# other mounts, and only the gateway network reaches it.
EXPOSE 8080
CMD ["python", "translator.py"]
