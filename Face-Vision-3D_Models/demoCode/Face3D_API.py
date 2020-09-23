from flask import Flask, request
import uuid
import pika
import paho.mqtt.client as mqtt
import glob

app = Flask(__name__)

@app.route('/genmodel', methods=['POST'])
def modelGen():
    path = str('../input/image-{}.jpg'.format(uuid.uuid1()))
    response = request.files
    image = response['image']
    image.save(path)
    client = mqtt.Client()
    client.connect("localhost")
    client.publish('process','yes')
    client.disconnect()
    f = open("testImages.txt", "w")
    n = f.write(path)
    f.close()
    return 'Image uploaded successfully'
    

if __name__ == "__main__":
    app.run(debug=True)