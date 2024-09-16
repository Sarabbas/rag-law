FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt 

COPY . .
EXPOSE 7000

#CMD ["python", "-m", "debugpy", "--listen", "0.0.0.0:5678", "--wait-for-client", "-m",]
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7000", "--reload"]
# CMD ["tail", "-f", "/dev/null"]