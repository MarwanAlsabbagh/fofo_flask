FROM python:3.10-slim

# تثبيت tesseract ودعم اللغة العربية
RUN apt-get update && \
    apt-get install -y tesseract-ocr tesseract-ocr-ara libglib2.0-0 libsm6 libxext6 libxrender-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# نسخ ملفات المشروع
WORKDIR /app
COPY . .

# تثبيت بايثون باكدج
RUN pip install --no-cache-dir -r requirements.txt

# تشغيل التطبيق
CMD ["python", "app.py"]
