import re, requests, os, random, time
from urllib.parse import unquote
# from langchain.text_splitter import RecursiveCharacterTextSplitter

from flask import Flask, jsonify, request, Response
import torch
import warnings
from transformers.utils import logging

import easyocr
reader = easyocr.Reader(['ar'], gpu=False)  # تأكد أن gpu=True فقط إذا كنت تستخدم GPU
            
##############################################################
            
app = Flask(__name__)

@app.route('/ocr')
def ocr():
    if 'image' not in request.files:
        return jsonify({'error': 'لم يتم إرسال صورة'}), 400

    # قراءة الصورة من الطلب
    image_file = request.files['image']
    image_path = 'uploaded_image.png'
    image_file.save(image_path)

    # استخدام EasyOCR لقراءة النصوص
    results = reader.readtext(image_path)
    for result in results:
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
