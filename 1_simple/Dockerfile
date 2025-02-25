FROM python:3.12-slim
WORKDIR /app
RUN pip install torch transformers
COPY app.py /app/app.py
CMD ["python", "app.py"]
