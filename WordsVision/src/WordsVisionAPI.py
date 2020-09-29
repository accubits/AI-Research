from flask import Flask, request, send_file
from inference import gen_img
from datetime import datetime
import PIL
from io import BytesIO

app = Flask(__name__)

def serve_pil_image(pil_img):
    img_io = BytesIO()
    pil_img.save(img_io, 'JPEG', quality=70)
    img_io.seek(0)
    return send_file(img_io, mimetype='image/jpeg')

@app.route('/wordsvision', methods=['POST'])
def genImage():
    response = request.json
    caption = response['caption']
    images = gen_img([caption])
    return serve_pil_image(images[2])

    

if __name__ == "__main__":
    app.run(debug=True)
