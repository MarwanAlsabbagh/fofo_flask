 from flask import Flask, jsonify, request, Response
 import os
 import requests as req
 import warnings
 from transformers.utils import logging

app = Flask(__name__)

@app.route('/test')
 def test():
     test = request.args["test"]
     return jsonify({'answer': 'OK' + test})


if __name__ == '__main__':
     app.run(debug=True, port=os.getenv("PORT", default=5000))
