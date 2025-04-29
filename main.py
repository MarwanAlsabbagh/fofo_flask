import re, requests, os, random, time
from urllib.parse import unquote
# from langchain.text_splitter import RecursiveCharacterTextSplitter

from flask import Flask, jsonify, request, Response
import torch
import warnings
from transformers.utils import logging

##############################################################
            
app = Flask(__name__)

def retrival():
    try:
        question = request.args["question"]
        answer=tune_question_answering(question)
        return jsonify({'question': question, 'answer': answer})
    except:
         return jsonify({'question': question, 'answer': "error server"})

if __name__ == '__main__':
    app.run(debug=True, port=os.getenv("PORT", default=5000))
