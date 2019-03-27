ARG TENSORFLOW_IMAGE=tensorflow/tensorflow:1.12.0-py3

FROM $TENSORFLOW_IMAGE

MAINTAINER pablo.lopezcoya@telefonica.com

WORKDIR /usr/src/app

ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8
ENV GIT_PYTHON_REFRESH quiet

# Install OpenCV libraries
RUN \
    apt-get install libglib2.0-0 -y && \
    apt-get install libsm6 -y && \
    apt-get install libxrender-dev -y && \
    apt-get install libxext6 -y

COPY Pipfile* ./

RUN pip3 install --upgrade pip  && \
    pip3 install pipenv && \
    pipenv install --system

# Download Spacy NLP models
RUN python3 -m spacy download en
RUN python3 -m spacy download es

COPY . ./

RUN mkdir logs
