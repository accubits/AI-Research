#!/bin/sh

/bin/bash -lc '
cd /home/ubuntu/AI-Research/WordsVision/
source venv/bin/activate
cd src/
uwsgi --ini wordsvision.ini'