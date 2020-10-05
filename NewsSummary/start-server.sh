#!/bin/sh

/bin/bash -lc '
cd /home/ubuntu/AI-Research/NewsSummary/
source venv/bin/activate
uwsgi --ini newssummary.ini'