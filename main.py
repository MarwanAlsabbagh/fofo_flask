import re, requests, os, random, time
from urllib.parse import unquote
# from langchain.text_splitter import RecursiveCharacterTextSplitter

from flask import Flask, jsonify, request, Response
import torch
import warnings
from transformers.utils import logging

import cv2
import numpy as np
from numpy.linalg import norm
from numpy import dot
from deepface import DeepFace

##############################################################
            
app = Flask(__name__)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def process_image(file):
    try:
        file.seek(0)
        img_bytes = file.read()
        img_array = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return None
            
@app.route('/verify', methods=['POST'])
def verify():
    try:
        if 'photo' not in request.files or 'stored' not in request.files:
            return jsonify({"status": "error", "message": "الرجاء إرسال صورتين"}), 400
        
        photo = request.files['photo']
        stored = request.files['stored']

        # معالجة الصور
        img_photo = process_image(photo)
        img_stored = process_image(stored)

        if img_photo is None or img_stored is None:
            return jsonify({"status": "error", "message": "صيغة الصورة غير صالحة"}), 400

        # الكشف عن الوجوه
        gray_photo = cv2.cvtColor(img_photo, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_photo, scaleFactor=1.1, minNeighbors=5)

        if len(faces) != 1:
            return jsonify({"status": "error", "message": "يجب أن تحتوي الصورة على وجه واحد فقط"}), 400

        # المقارنة باستخدام DeepFace
        try:
            result = DeepFace.verify(img_photo, img_stored, model_name='Facenet', detector_backend='opencv')
            return jsonify({
                "status": "success",
                "verified": result["verified"],
                "similarity": float(result["distance"])
            })
        except Exception as e:
            return jsonify({"status": "error", "message": f"فشل في المقارنة: {str(e)}"}), 500

    except Exception as e:
        return jsonify({"status": "error", "message": f"خطأ غير متوقع: {str(e)}"}), 500

def retrival():
    try:
        question = request.args["question"]
        answer=tune_question_answering(question)
        return jsonify({'question': question, 'answer': answer})
    except:
         return jsonify({'question': question, 'answer': "error server"})

if __name__ == '__main__':
    app.run(debug=True, port=os.getenv("PORT", default=5000))
