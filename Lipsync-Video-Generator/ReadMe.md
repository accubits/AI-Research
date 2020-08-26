# Lipsync-Video-Generator

Generates a video with facial movements from an input of source image and a driving video. 
Sasi Tharoor and Donald Trumps images are given in the `input_images` folder. 

### Dependencies

required packages: 

- imageio==2.3.0
- matplotlib==2.2.2
- numpy==1.15.0
- pandas==0.23.4
- scipy==1.1.0
- toolz==0.9.0
- torch==1.0.0
- torchvision==0.2.1

run `pip install -r requirements.txt` to install the required dependencies.

A default video input is givin in the `input_video` folder. 
To use a different video run `python3 crop-video.py --inp path/to/video.mp4` to generate a cropped input video. This will give a cropping suggestion and cropps the video. 


### Inference
- Run the command below from `code/` directory to generate the video. The results saved in the `output_video` directory.
Choice = 0 for Sasi Tharoor and 1 for Donald Trump
```
python3 inference.py --person CHOICE 
```


### Pretrained Model
- Please download models from [here](https://drive.google.com/file/d/1_v_xW1V52gZCZnXgh1Ap_gwA9YVIzUnS/view?usp=sharing) and put it under `code/`.

### Sample Outputs

<img src="readme_files/output.gif"/>