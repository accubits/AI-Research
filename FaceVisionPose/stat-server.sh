#!/bin/sh

/bin/bash -lc '
cd /home/ubuntu/AI-Research/FaceVisionPose/
source venv/bin/activate
python PoseQueue.py
uwsgi --ini pose.ini'