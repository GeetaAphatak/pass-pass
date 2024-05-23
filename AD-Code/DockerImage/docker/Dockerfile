# Build an image that can used in Anomaly Detection
FROM ubuntu:18.04

RUN apt-get -y update && apt-get install -y --no-install-recommends \
         wget \
         python \
         nginx \
         default-jre \
         libenchant1c2a \
         libmysqlclient-dev \
         ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update \
	&& apt-get install -y python3-pip python3-dev \
	&& cd /usr/local/bin \
	&& ln -s /usr/bin/python3 python \
	&& pip3 install --upgrade pip

# Inorder to run pip3.6, run it the following way `python3.6 -m pip`
# Update Python & Install wheel
RUN python3.6 -m pip install pip --upgrade && \
    python3.6 -m pip install wheel

COPY requirements.txt ./
RUN pip install -r requirements.txt

ENV PYTHONUNBUFFERED=TRUE \
   PYTHONDONTWRITEBYTECODE=TRUE \
   LANG=C.UTF-8 \
   LC_ALL=C.UTF-8
