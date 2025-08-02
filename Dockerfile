FROM python:3.10-slim

WORKDIR /app

COPY . .
COPY model.joblib artifacts/model.joblib
RUN pip install --upgrade pip && pip install -r requirements.txt

CMD ["python", "src/predict.py"]
