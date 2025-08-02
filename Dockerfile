FROM python:3.10-slim

WORKDIR /app

# Copy everything (including src/, utils.py, train.py, requirements.txt)
COPY . .

# Upgrade pip and install dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Train the model and save it inside the container
RUN python -m src.train

# Run prediction script by default
CMD ["python", "-m", "src.predict"]
