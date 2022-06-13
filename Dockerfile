FROM ubuntu:latest

RUN apt update

RUN apt-get install python3 python3-pip -y

RUN mkdir -p numerous

COPY requirements.txt ./numerous

WORKDIR numerous

RUN pip3 install -r requirements.txt

