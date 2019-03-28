ARG TENSORFLOW_IMAGE=tensorflow/tensorflow:1.12.0-py3
FROM $TENSORFLOW_IMAGE

MAINTAINER pablo.lopezcoya@telefonica.com

WORKDIR /home/gymnos

ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
ENV GIT_PYTHON_REFRESH quiet
ENV KAGGLE_USERNAME ""
ENV KAGGLE_KEY ""
ENV TF_CPP_MIN_LOG_LEVEL 2  # supress tensorflow info output, only logs errors

# Install OpenCV libraries
RUN \
    apt-get update && \
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

# Save keras cache into gymnos cache
RUN mkdir /root/.keras/ /home/gymnos/cache
RUN ln -s /root/.keras/ /home/gymnos/cache/keras

VOLUME ["/home/gymnos/cache"]

COPY src ./

RUN mkdir logs

ENTRYPOINT ["python3", "-m", "bin.scripts.gymnosd"]
