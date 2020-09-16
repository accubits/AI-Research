FROM ubuntu:18.04

RUN apt-get update -y && apt-get install software-properties-common -y

RUN apt-get update
RUN apt-get install -y build-essential python3 python3-dev python3-pip
RUN python3 -m pip install pip --upgrade && \
        python3 -m pip install wheel

RUN mkdir scrape

COPY ./scrape_summarize.py /scrape
COPY ./scrapper_api.py /scrape
COPY ./requirements.txt /scrape
COPY ./run.sh /scrape

WORKDIR /scrape

RUN pip install -r requirements.txt

RUN chmod +x run.sh

ENTRYPOINT ["bash", "run.sh"]
