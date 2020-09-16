from flask import Flask, request
from datetime import datetime

import nn_process
import time
import os
import sys

from PIL import Image

print ('Loading Extracting Feature Module...')
extract_feature = nn_process.create('extract_feature')
print ('Loading Generating Poem Module...')
generate_poem = nn_process.create('generate_poem')
DEFAULT_PATH = ''

app = Flask(__name__)

@app.route('/generate', methods=['POST'])
def genImage():
    response = request.files
    image = Image.open(response['image'])
    img_feature = extract_feature(image)
    poem = generate_poem(img_feature)
    return poem[0]
    
if __name__ == "__main__":
    app.run(debug=False)
