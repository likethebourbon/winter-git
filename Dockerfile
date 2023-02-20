FROM python:3.10-slim

ENV PYTHONUNBUFFERED True

ENV PATH="/venv/bin:$PATH"

COPY . ./

RUN pip install --no-cache-dir -r requirements.txt

CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app:application