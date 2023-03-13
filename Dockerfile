FROM python:3.8-slim-buster

RUN apt-get update \
  && apt-get install -y wget \
  && rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install -y git
RUN git clone https://github.com/gradient-ai/llama
WORKDIR llama/
RUN pip install -r llama/requirements.txt

COPY 7B/* ./ 