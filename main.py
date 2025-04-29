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

##############################################################
            
app = Flask(__name__)

# تحميل نموذج Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces

def check_liveness(faces, img_photo, img_stored):
    if len(faces) != 1:
        return {"status": "error", "message": "لا توجد وجوه حقيقية أو صورة مزيفة ❌"}
    else:
        # (هنا يمكن إضافة المزيد من الفحوص مثل تحليل الحركة أو الوميض)
        # تحويل الصور إلى numpy arrays
        img_photo = np.frombuffer(img_photo.read(), np.uint8)
        img_stored = np.frombuffer(img_stored.read(), np.uint8)
        
        img_photo = cv2.imdecode(img_photo, cv2.IMREAD_COLOR)
        img_stored = cv2.imdecode(img_stored, cv2.IMREAD_COLOR)
        
        # استخراج الميزات (يحتاج لتثبيت مكتبة face_recognition أو deepface)
        # مثال باستخدام DeepFace:
        from deepface import DeepFace
        try:
            result = DeepFace.verify(img_photo, img_stored, model_name='Facenet', detector_backend='opencv')
            if result['verified']:
                return {"status": "success", "message": "نفس الشخص ✅"}
            else:
                return {"status": "error", "message": "أشخاص مختلفين ❌"}
        except:
            return {"status": "error", "message": "لم يتم الكشف عن وجه في واحدة من الصور"}
            
@app.route('/verify', methods=['POST'])
def verify():
    if 'photo' not in request.files or 'stored' not in request.files:
        return jsonify({"error": "الرجاء إرسال صورتين"})
    
    photo = request.files['photo']
    stored = request.files['stored']
    
    faces = detect_face(cv2.imdecode(np.frombuffer(photo.read(), np.uint8), cv2.IMREAD_COLOR)
    result = check_liveness(faces, photo, stored)
    
    return jsonify(result)

# @app.route('/retrival')
def retrival():
    try:
        question = request.args["question"]
        answer=tune_question_answering(question)
        return jsonify({'question': question, 'answer': answer})
    except:
         return jsonify({'question': question, 'answer': "error server"})

if __name__ == '__main__':
    app.run(debug=True, port=os.getenv("PORT", default=5000))
