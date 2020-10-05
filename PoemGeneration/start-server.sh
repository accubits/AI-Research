#!/bin/sh

/bin/bash -lc '
cd /home/ubuntu/AI-Research/PoemGeneration/
source venv/bin/activate
cd src/
uwsgi --ini poem.ini'