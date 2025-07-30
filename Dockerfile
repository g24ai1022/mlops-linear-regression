FROM python:3.10-slim

WORKDIR /app

COPY . .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Train the model and save it inside the container
RUN python src/train.py

CMD ["python", "src/predict.py"]
