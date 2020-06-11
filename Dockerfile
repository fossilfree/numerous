FROM ubuntu:latest

RUN apt-get install python3 python3-pip -y

RUN mkdir -p numerous

WORKDIR numerous

COPY requirments.txt ./

RUN pip install -r requirements.txt
