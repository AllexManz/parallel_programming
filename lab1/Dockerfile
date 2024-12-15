FROM alpine:latest

RUN apk update
RUN apk add --no-cache gcc g++ make musl-dev libgomp openmpi openmpi-dev
RUN apk add git

RUN mkdir /home/lab1
RUN cd /home/lab1

COPY . /Lab1
WORKDIR /Lab1

LABEL authors="alex"

RUN gcc -fopenmp -o par_prog_lab1 lab1.c
CMD ["./par_prog_lab1"]