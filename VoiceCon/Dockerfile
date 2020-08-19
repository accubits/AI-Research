FROM tiangolo/uwsgi-nginx-flask:python3.7

WORKDIR /workspace
RUN apt-get clean \
        && apt-get update \
        && apt-get install -y ffmpeg libportaudio2 openssh-server python3-pyqt5 xauth \
        && apt-get -y autoremove

ADD requirements.txt /workspace/requirements.txt
RUN pip install -r /workspace/requirements.txt


EXPOSE 5000
ENTRYPOINT [ "python" ] 
CMD [ "app.py" ] 
