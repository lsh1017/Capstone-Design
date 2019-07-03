import os
from kerasScript import classification
from flask import Flask, request, jsonify

UPLOAD_FOLDER = './uploads'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['JSON_AS_ASCII'] = False


@app.route('/')
def image_api_server():
    return 'image_api_server'

@app.route('/image', methods=['POST'])
def classification_image():
    if 'file' not in request.files:
        return jsonify(success=0, message='not existed file')
    else:
        image = request.files['file']

        if image.filename == '':
            return jsonify(success=0, message='no file detected')
        if image:
            imageName = image.filename
            fileName = os.path.join(app.config['UPLOAD_FOLDER'], imageName)
            image.save(fileName)
            result = classification(fileName)
            return jsonify(result)

    return jsonify(success=0, message='error')


if __name__ == '__main__':
    app.run(host='0.0.0.0')
