FROM python:3.8.13-buster

COPY requirements_api.txt /requirements.txt

RUN pip install -U pip
RUN pip install -r requirements.txt

COPY api /api
COPY artist_studies /artist_studies

CMD uvicorn api.fast:app --host 0.0.0.0 --port $PORT
