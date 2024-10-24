FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt ./requirements.txt
RUN apt-get update \
    && apt-get -y install libpq-dev gcc \
    && pip install psycopg2
# Install uvicorn
RUN pip install uvicorn
# Install dependencies
RUN pip install -r requirements.txt
COPY . /app
ENTRYPOINT ["uvicorn", "main:app"]
CMD ["--host", "0.0.0.0", "--port", "7860"]