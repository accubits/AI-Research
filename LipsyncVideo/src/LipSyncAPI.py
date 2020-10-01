from flask import Flask, request, send_file
from datetime import datetime
import uuid
import paho.mqtt.client as mqtt

app = Flask(__name__)

@app.route('/lipsyncgen', methods=['POST'])
def lipsync():
    path_image = str('../input_images/API/image-{}.jpg'.format(uuid.uuid1()))
    path_video = str('../input_video/API/video-{}.mp4'.format(uuid.uuid1()))
    response = request.files
    image = response['image']
    video = response['video']
    image.save(path_image)
    video.save(path_video)
    client = mqtt.Client()
    client.connect("localhost")
    client.publish('filenames','{}|{}'.format(path_image,path_video))
    client.disconnect()
    return 'Image uploaded successfully'
    
if __name__ == "__main__":
    app.run(debug=True)
