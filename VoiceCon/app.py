from flask import Flask, request
from flask_cors import CORS
from router import Router


app = Flask(__name__)
app.config.from_pyfile('config.py')
app.static_folder = app.config['STATIC_DIR']
CORS(app)

router = Router(app.config)


@app.route('/blob/<path:path>')
def send_result(path):
    return app.send_static_file(path)


@app.route("/embed", methods=["POST"])
def create_embeddings():
    result, status = router.audio2embeddings(request.files['audio'], request.form['speaker'], request.form.get('seq_length'))
    return {"success": status, "result": result}, 200 if status else 422


@app.route("/generate", methods=["POST"])
def generate_audio():
    result, status = router.text2speech(request.form.get('embed_id'), request.form.get('texts'))
    return {"success": status, "result": result}, 200 if status else 422


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)