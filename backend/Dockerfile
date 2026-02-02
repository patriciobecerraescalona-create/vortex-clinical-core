FROM python:3.12-slim

WORKDIR /workspace

RUN pip install --no-cache-dir -U pip

COPY backend/requirements.txt /workspace/backend/requirements.txt
RUN pip install --no-cache-dir -r /workspace/backend/requirements.txt

EXPOSE 8000

CMD ["uvicorn", "backend.app.main:app", "--host=0.0.0.0", "--port=8000", "--reload"]
