FROM python:3.10-slim

WORKDIR /app

# Copy everything (including src/, utils.py, train.py, requirements.txt)
COPY . .

# Upgrade pip and install dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Train the model and save it inside the container
# Use module run to avoid import errors (make sure __init__.py exists in src/)
RUN python -m src.train

# Run prediction script by default
CMD ["python", "-m", "src.predict"]
