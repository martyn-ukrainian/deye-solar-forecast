# Живий прогноз сонця на Pi: 4 моделі щогодини → live/solar.db. Моделі — з train.py (models/).
FROM python:3.13-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements-pi.txt .
RUN pip install --no-cache-dir -r requirements-pi.txt
COPY *.py ./
COPY models/ models/
ENV PYTHONUNBUFFERED=1
CMD ["python", "live.py"]
