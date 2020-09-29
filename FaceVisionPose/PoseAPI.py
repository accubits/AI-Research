from flask import Flask, request
from datetime import datetime
import uuid
import paho.mqtt.client as mqtt

app = Flask(__name__)

@app.route('/genpose', methods=['POST'])
def poseGen():
    path = str('./input/image-{}.jpg'.format(uuid.uuid1()))
    response = request.files
    image = response['image']
    image.save(path)
    client = mqtt.Client()
    client.connect("localhost")
    client.publish('filename',path)
    client.disconnect()
    f = open("testImages.txt", "w")
    n = f.write(path)
    f.close()
    return 'Image uploaded successfully'
    

if __name__ == "__main__":
    app.run(debug=True)
