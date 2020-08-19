#!/bin/bash
app="docker.voicecon"
docker run -it --rm --init -p 5000:5000 --name=${app} -v $PWD:/workspace ${app}
