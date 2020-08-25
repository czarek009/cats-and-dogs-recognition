FROM ubuntu:20.04

WORKDIR /cats-vs-dogs

RUN apt-get update
RUN apt-get install -y python3 python3-pip

RUN pip3 install numpy
RUN pip3 install tensorflow
RUN pip3 install opencv-python