FROM python:3.10-slim

WORKDIR /app

# Ставимо libsndfile для роботи soundfile
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Встановлюємо тільки потрібні залежності
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Копіюємо тільки код проєкту
COPY . .

EXPOSE 5000

CMD ["python", "app.py"]
