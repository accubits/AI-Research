from flask import Flask, request, send_file
from datetime import datetime

from inference import load_checkpoints,random_input,make_animation
from PIL import Image
import torch
import json
import numpy as np
import imageio
from skimage.transform import resize
from skimage import img_as_ubyte
import cv2
from io import BytesIO
import base64

app = Flask(__name__)

@app.route('/lipsync', methods=['POST'])
def genImage():
    response = request.files
    image = response['image']
    video = response['video']

    video.save('../input_video/crop.mp4')

    source_image = np.array(Image.open(image))

    output_path = '../output_video/temp.gif'

    reader = imageio.get_reader('../input_video/crop.mp4')

    fps = reader.get_meta_data()['fps']
    driving_video = []
    try:
        for im in reader:
            driving_video.append(im)
    except RuntimeError:
        pass
    reader.close()

    source_image = resize(source_image, (256, 256))[..., :3]
    driving_video = [resize(frame, (256, 256))[..., :3] for frame in driving_video]
    generator, kp_detector = load_checkpoints(config_path='vox-256.yaml', checkpoint_path='vox-cpk.pth.tar', cpu=False)

    predictions = make_animation(source_image, driving_video, generator, kp_detector, relative=True, adapt_movement_scale=True, cpu=False)
    imageio.mimsave(output_path, [img_as_ubyte(frame) for frame in predictions], fps=fps)  

    return send_file(output_path, mimetype='image/gif')
    
if __name__ == "__main__":
    app.run(debug=True)
