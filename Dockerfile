FROM tensorflow/tensorflow

MAINTAINER pablo.lopezcoya@telefonica.com



RUN { \
    echo '[Ubuntu-base]'; \
    echo 'VERSION="16.04.5 LTS (Xenial Xerus)"'; \
    echo 'ID=ubuntu'; \
    echo 'ID_LIKE=debian'; \
    echo 'PRETTY_NAME="Ubuntu 16.04.5 LTS"'; \
    echo 'VERSION_ID="16.04"';} \
    && apt-get update


RUN apt-get install tree -y \
    && apt-get install apt-utils -y \
    && apt-get install vim -y \
    && apt-get install libsm6 libxext6 -y \ 
    && apt-get install libfontconfig1 libxrender1 -y \
    && apt-get install python-pip -y \
    && pip install --upgrade pip \
    && pip install tensorflow \
    && pip install keras \
    && pip install opencv-python \
    && pip install progressbar

ENV PYTHONPATH=":/home/sysadmin/aitp/lib\
:/home/sysadmin/aitp/lib/models\
:/home/sysadmin/aitp/lib/services\
:/home/sysadmin/aitp/lib/datasets\
:/home/sysadmin/aitp/lib/var"

RUN useradd -d /home/sysadmin -ms /bin/bash -g root -p sysadmin sysadmin

COPY . /home/sysadmin/aitp/
RUN mkdir -p /home/sysadmin/aitp/logs \
    && mkdir -p /home/sysadmin/aitp/datasets \
    && chmod 777 -R /home/sysadmin/aitp/

#USER sysadmin
USER root
WORKDIR /home/sysadmin
