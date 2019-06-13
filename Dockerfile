FROM tensorflow/tensorflow:1.13.1-py3

MAINTAINER pablo.lopezcoya@telefonica.com

WORKDIR /home/gymnos

ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

COPY Pipfile* ./

RUN pip3 install --upgrade pip  && \
    pip3 install pipenv && \
    pipenv install --system

COPY src ./

ENV GIT_PYTHON_REFRESH quiet

ENTRYPOINT /bin/bash
