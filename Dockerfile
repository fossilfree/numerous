FROM ubuntu:latest

RUN echo "deb http://archive.ubuntu.com/ubuntu bionic main universe" >> /etc/apt/sources.list &&\
	echo "deb http://archive.ubuntu.com/ubuntu bionic-security main universe" >> /etc/apt/sources.list &&\
	echo "deb http://archive.ubuntu.com/ubuntu bionic-updates main universe" >> /etc/apt/sources.list
RUN apt update

RUN apt-get install python3 python3-pip -y

RUN mkdir -p numerous

COPY requirements.txt ./numerous

WORKDIR numerous

RUN pip3 install -r requirements.txt

