FROM python:3.8.13-buster

COPY requirements_api.txt /requirements.txt

RUN pip install -U pip
RUN pip install -r requirements.txt
