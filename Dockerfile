# 1. Base image: Python 3.10 slim version use kar rahe hain taaki build fast aur light ho
FROM python:3.10-slim

# 2. Environment variables set kar rahe hain taaki Python logs turant dikhein
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 3. Working directory create karo
WORKDIR /app

# 4. Pehle requirements copy karke install karo (Docker caching ka fayda milega)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Baaki sara code copy karo
COPY . .

# 6. Hugging Face Spaces default port 7860 use karta hai
EXPOSE 7860

# 7. Server ko start karne ki command
# Hum server/main.py ko uvicorn ke saath chala rahe hain
# Purani line ko hata kar ye dalo:
CMD ["python", "-u", "app.py"]