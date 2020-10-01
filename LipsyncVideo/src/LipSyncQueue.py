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
import paho.mqtt.client as mqtt

def on_connect(client, userdata, flags, rc):
    print("Connected with result code "+str(rc))
    client.subscribe("filenames")

def on_message(client, userdata, msg):
	filenames = msg.payload.decode()
	run(filenames)
    # client.disconnect()

def run(filenames):
    image = filenames.split('|')[0]
    video = filenames.split('|')[1]
    source_image = np.array(Image.open(image))

    output_path = '../output_video/{}.gif'.format(image.split('/')[-1].split('.')[0])

    reader = imageio.get_reader(video)

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
    generator, kp_detector = load_checkpoints(config_path='vox-256.yaml', checkpoint_path='vox-cpk.pth.tar', cpu=True)

    predictions = make_animation(source_image, driving_video, generator, kp_detector, relative=True, adapt_movement_scale=True, cpu=True)
    imageio.mimsave(output_path, [img_as_ubyte(frame) for frame in predictions], fps=fps)  

    return None

if __name__ == "__main__":
    client = mqtt.Client()
    client.connect("localhost")

    client.on_connect = on_connect
    client.on_message = on_message

    client.loop_forever()
