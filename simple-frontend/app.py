import base64
import time
import requests

from flask import Flask, request, send_file, render_template, flash

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    try:
        data = request.files['image']
        f = data.read()
        encoded_data = base64.b64encode(f)
        r = requests.get('http://infer/api/v1/infer', params={'image': encoded_data.decode(encoding='utf8')})
        if r.status_code == 200:
            res = r.json['result']
            msg = f'{res}이 인식되었습니다!'
        else:
            msg = '내부 오류가 발생했습니다. 잠시 후에 다시 시도해 주세요. (고객센터: 02-872-5127)'
        render_template('index.html', messages=msg)
    except KeyError:
        # image field not provided
        return 'bad request', 400
    # expose server error


if __name__ == "__main__":
  app.run(host='0.0.0.0', port=80)
