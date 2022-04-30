# syntax=docker/dockerfile:1
FROM eitanhemed/robusta:latest

LABEL maintainer="Eitan.Hemed@gmail.com"

RUN mkdir home/robusta-dev
ADD ./robusta home/robusta-dev/
RUN cd home/robusta-dev
VOLUME /output
 