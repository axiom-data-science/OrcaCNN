# syntax=docker/dockerfile:1

FROM python:3.8
WORKDIR /app

ADD . /app

RUN apt update && apt upgrade -y && apt install -y python3 python3-pip libsndfile1 

RUN --mount=type=cache,id=custom-pip,target=/root/.cache/pip pip3 install -r requirements.txt

CMD ["uwsgi", "app.ini"]