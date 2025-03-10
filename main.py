from flask import Flask, jsonify, request, Response
import os
import requests as req

app = Flask(__name__)

@app.route('/chatbot1')
def chatbot1():
    question = request.args["question"]
    answer="ok"
    return jsonify({'question': question, 'answer': answer})

@app.route('/chatbot2')
def chatbot2():
    question = request.args["question"]
    answer="ok"
    return jsonify({'question': question, 'answer': answer})

if __name__ == '__main__':
    app.run(debug=True, port=os.getenv("PORT", default=5000))
