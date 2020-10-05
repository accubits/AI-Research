#!/bin/sh

/bin/bash -lc '
cd /home/ubuntu/AI-Research/LipsyncVideo/
source venv/bin/activate
cd src/
python lipSyncQueue.py
uwsgi --ini lipsync.ini'