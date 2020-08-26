FROM tensorflow/tensorflow:latest-gpu-py3

RUN apt-get update

RUN pip3 install opencv-python