FROM python:3.10-slim

WORKDIR /app

COPY . .
RUN pip install --upgrade pip && pip install -r requirements.txt
COPY model.joblib models/model.joblib

CMD ["python", "src/predict.py"]
