from flask import Flask, render_template, request,send_file
import paho.mqtt.client as mqtt
from src.gpt_inference import inference as gpt_inference
from src.storyGAN_inference import inference as storyGAN_inference
import os

app = Flask(__name__)

@app.route('/gpt_train', methods=['POST'])
def train_gpt():
    file = request.files['file']
    filename = file.filename
    
    if filename.split('.')[-1] == 'txt':
        if os.path.exists('src/training_data'):
            pass
        else:
            os.mkdir('src/training_data')
        file.save('src/training_data/'+filename)
        client = mqtt.Client()
        client.connect('localhost')
        client.publish('filename',filename)
        client.disconnect()
        return 'Training Initialized'
    else:
        return 'Please upload .txt file'

@app.route('/gpt_sample', methods=['POST'])
def sample_gpt():
    text = request.json['input']
    output = gpt_inference(input_text=text)
    return output

@app.route('/storygan_sample', methods=['POST'])
def sample_storygan():
    text = request.json['input']
    image = storyGAN_inference(text)
    return send_file('src/temp/out.gif',mimetype='image/gif')


if __name__ == '__main__':
    app.run(debug=True)