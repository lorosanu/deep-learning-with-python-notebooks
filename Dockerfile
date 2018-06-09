FROM ubuntu:16.04
MAINTAINER lorosanu@users.noreply.github.com

ENV \
  LANG=C.UTF-8 \
  TZ=Europe/Paris \
  MPLBACKEND=agg

RUN apt-get update \
  && apt-get install -y python3 python3-pip python3-tk \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/* \
  && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN pip3 install \
  numpy==1.14.4 \
  scipy==1.1.0 \
  matplotlib==2.2.0 \
  ipython==6.4.0 \
  notebook==5.5.0 \
  tensorflow==1.5.0 \
  keras==2.1.3

CMD python3
