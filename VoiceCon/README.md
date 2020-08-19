# VoiceCon

VoiceCon is a production ready implementation of a three-stage deep learning framework that allows to create a numerical representation of a voice from a few seconds of audio, and to use it to condition a text-to-speech model trained to generalize to new voices.

<br>

# Deployment

### Clone the Repo:
  
  - git clone https://github.com/accubits/AI-Research.git
  - cd AI-Research/VoiceCon    

<br>

### Download pre-trained models:
  
  - [Google Drive](https://drive.google.com/file/d/1n1sPXvT34yXFLT47QZA6FIRGrwMeSsZc/view?usp=sharing)
  - Extract the zip file
    
    <br>
    
    ```sh
    unzip pretrained.zip
    ```
  
<br>
  
### Docker setup:
  
  - Run build.sh file to build the docker image
    
    <br>
  
    ```sh
    chmod +x build.sh
    ./build.sh
    ```
    <br>
    
  - To start the dockerized API, run start.sh
    
    <br>
    
    ```sh
    chmod +x start.sh
    ./start.sh
    ```    

<br>

# API Docs

 - [Postman collection](https://www.getpostman.com/collections/17e2fc0795fbff6aa9b4)
 - [Postman Docs](https://documenter.getpostman.com/view/8991468/T1LSCRas)
 
<br> 

# Results

 - [Dr. Shashi Tharoor](https://github.com/accubits/AI-Research/blob/master/VoiceCon/results/Accubits_shashi_tharoor.wav)
 - [Donald Trump](https://github.com/accubits/AI-Research/blob/master/VoiceCon/results/Accubits_trump.wav)
