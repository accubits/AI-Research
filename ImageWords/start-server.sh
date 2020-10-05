#!/bin/sh

/bin/bash -lc '
cd /home/ubuntu/AI-Research/ImageWords/
source venv/bin/activate
uwsgi --ini imagewords.ini'