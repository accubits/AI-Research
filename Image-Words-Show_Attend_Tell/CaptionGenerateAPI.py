from flask import Flask, request
from datetime import datetime

from caption import caption_image_beam_search
from PIL import Image
import torch
import json
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = 'model_files/BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar'
word_map = 'model_files/WORDMAP_coco_5_cap_per_img_5_min_word_freq.json'

# Load model
checkpoint = torch.load(model, map_location=str(device))
decoder = checkpoint['decoder']
decoder = decoder.to(device)
decoder.eval()
encoder = checkpoint['encoder']
encoder = encoder.to(device)
encoder.eval()

# Load word map (word2ix)
with open(word_map, 'r') as j:
    word_map = json.load(j)
rev_word_map = {v: k for k, v in word_map.items()}  # ix2word

app = Flask(__name__)

@app.route('/captiongen', methods=['POST'])
def genImage():
    response = request.files
    image = np.array(Image.open(response['image']))
    seq, alphas = caption_image_beam_search(encoder, decoder, image, word_map, 5)
    words = [rev_word_map[ind] for ind in seq]
    output = ''
    for i in words[1:-1]:
        output += ' {}'.format(i)
    return output
    
if __name__ == "__main__":
    app.run(debug=False)
