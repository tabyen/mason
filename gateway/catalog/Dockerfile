FROM python:3.13-slim
WORKDIR /app
COPY catalog.py .
COPY models.toml /data/models.toml
RUN useradd -r app
USER app
EXPOSE 8090
CMD ["python", "catalog.py"]
