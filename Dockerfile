FROM runpod/pytorch:1.0.3-cu1290-torch260-ubuntu2204

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY handler.py .

CMD ["python", "-u", "handler.py"]
