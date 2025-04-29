import re, requests, os, random, time
from urllib.parse import unquote
# from langchain.text_splitter import RecursiveCharacterTextSplitter

from flask import Flask, jsonify, request, Response
import torch
import warnings
from transformers.utils import logging

from PIL import Image
import pytesseract
##############################################################
            
app = Flask(__name__)


@app.route('/ocr', methods=['POST'])
def ocr():
    if 'image' not in request.files:
        return jsonify({'error': 'لم يتم إرسال صورة'}), 400

    image_file = request.files['image']
    image_path = 'uploaded_image.png'
    image_file.save(image_path)

    try:
        # تحميل الصورة
        img = Image.open(image_path)

        # استخراج النص (مع تحديد اللغة العربية)
        text = pytesseract.image_to_string(img, lang='ara')

        return jsonify({'text': text.strip()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500:
        print(result[1])

def retrival():
    try:
        question = request.args["question"]
        answer=tune_question_answering(question)
        return jsonify({'question': question, 'answer': answer})
    except:
         return jsonify({'question': question, 'answer': "error server"})

if __name__ == '__main__':
    app.run(debug=True, port=os.getenv("PORT", default=5000))
