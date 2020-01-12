from flask import Flask, request
from flask.templating import render_template
from flask.wrappers import Response

from stream_model import Model

model = Model()

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict_article():
    labels = model.predict(request.form['text'])
    return Response("<br>".join(labels))


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
