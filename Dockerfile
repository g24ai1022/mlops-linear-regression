FROM python:3.10-slim

WORKDIR /app

COPY . .

RUN pip install --upgrade pip && pip install -r requirements.txt

ENV PYTHONPATH=/app

RUN python -m src.train

CMD ["python", "-m", "src.predict"]
