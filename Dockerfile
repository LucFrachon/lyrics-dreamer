FROM python:3.9-slim-buster
WORKDIR /app
ADD . /app
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8080
CMD ["python", "app.py"]
