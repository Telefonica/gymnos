FROM python:3.6

MAINTAINER pablo.lopezcoya@telefonica.com

WORKDIR /home/gymnos

ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

COPY setup.py ./
COPY README.md ./

COPY gymnos ./gymnos
COPY scripts ./scripts
COPY experiments ./experiments

RUN pip3 install --upgrade pip  && \
    pip3 install -e . && \
    pip3 install .[tensorflow]

ENTRYPOINT /bin/bash
