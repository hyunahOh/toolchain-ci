import base64
import time

from flask import Flask, request
import numpy as np
from backend.test import detect_img
from PIL import Image

app = Flask(__name__)


@app.route('/')
def index():
    return str(['/api/v1/infer'])


@app.route('/api/v1/infer')
def upload():
    try:
        data = request.args.get('image')
        image = Image.open(base64.b64decode(data))
        return str({'result': detect_img(np.array(image)[...,:3])})
    except KeyError:
        # image field not provided
        return 'bad request', 400
    # expose server error


if __name__ == "__main__":
  app.run(host='0.0.0.0', port=80)
