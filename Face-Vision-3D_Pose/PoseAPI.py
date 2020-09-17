from flask import Flask, request
from main import main
from datetime import datetime

app = Flask(__name__)

@app.route('/genpose', methods=['POST'])
def poseGen():
    response = request.files
    image = response['image']
    main(image)
    return 'gg'
    

if __name__ == "__main__":
    app.run(debug=False)
