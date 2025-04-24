import re, requests, os, random, time, html
from langchain_community.retrievers import BM25Retriever
from bs4 import BeautifulSoup
from urllib.parse import unquote
from langchain.text_splitter import RecursiveCharacterTextSplitter
from flask import Flask, jsonify, request, Response
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from numpy import dot, norm
import logging
import torch
import warnings
from transformers.utils import logging

# تهيئة Flask App
app = Flask(__name__)

# ---------------------- جزء التحقق من الوجوه ----------------------
face_app = FaceAnalysis(name="buffalo_l")
face_app.prepare(ctx_id=0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
logging.basicConfig(level=logging.INFO)

def detect_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# ---------------------- جزء المعالجة اللغوية ----------------------
download_dir = "./download_dir_rag"
os.makedirs(download_dir, exist_ok=True)
pages = []
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=80058,
    chunk_overlap=5000,
    is_separator_regex=True,
    separators=["\ufeff"]
)

# ---------------------- وظائف تنزيل الملفات ----------------------
def down(url):
    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            content_disposition = response.headers.get('Content-Disposition', '')
            filename = unquote(content_disposition.split('filename=')[1].split(';')[0].strip('"\'')).encode('latin-1').decode('utf-8') if 'filename=' in content_disposition else "file.txt"
            
            file_path = os.path.join(download_dir, filename)
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=100):
                    if chunk: f.write(chunk)
            
            with open(file_path, 'rb') as f:
                raw_content = f.read()
                content = raw_content.decode('utf-8', errors='replace').replace("‏","").replace("بشار الأسد","").replace('\n\n', ' ')
            
            pages.append({'content': content, 'title': filename})
    except Exception as e:
        print(f"Error downloading {url}: {e}")

# ---------------------- تهيئة النظام عند التشغيل ----------------------
@app.before_first_request
def initialize_system():
    # تنزيل الملفات القانونية
    url = "https://groups.google.com/g/syrianlaw/c/Wba7S8LT9MU?pli=1"
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        urls = [tag.get("href") for tag in soup.find_all("a")]
        for url in urls[::3]:
            if url and "https://docs" in url:
                down(url)
    
    # معالجة النصوص
    global retriever
    final_pages = [p for p in pages if len(p['content']) <= 6500]
    texts = text_splitter.create_documents([p['content'].replace('\ufeff', '') for p in final_pages])
    retriever = BM25Retriever.from_texts(
        texts=[doc.page_content for doc in texts],
        ngram_range=(2, 2),
        k=3
    )

# ---------------------- الروتات الأساسية ----------------------
@app.route('/Face_Verification', methods=['POST'])
def verify_faces():
    try:
        if 'photo' not in request.files or 'stored' not in request.files:
            return jsonify({"success": False, "message": "يجب إرسال ملفين باسم 'photo' و 'stored'"}), 400
        
        photo_img = cv2.imdecode(np.frombuffer(request.files['photo'].read(), np.uint8), cv2.IMREAD_COLOR)
        stored_img = cv2.imdecode(np.frombuffer(request.files['stored'].read(), np.uint8), cv2.IMREAD_COLOR)
        
        # التحقق من عدد الوجوه
        if len(detect_face(photo_img)) != 1 or len(detect_face(stored_img)) != 1:
            return jsonify({"success": False, "message": "يجب أن تحتوي كل صورة على وجه واحد فقط"}), 400
        
        # استخراج الميزات
        emb1 = face_app.get(photo_img)[0].embedding
        emb2 = face_app.get(stored_img)[0].embedding
        similarity = dot(emb1, emb2)/(norm(emb1)*norm(emb2))
        
        return jsonify({
            "success": True,
            "similarity": float(similarity),
            "is_match": similarity > 0.5,
            "message": "نفس الشخص ✅" if similarity > 0.5 else "أشخاص مختلفين ❌"
        })
    except Exception as e:
        app.logger.error(f"Error: {str(e)}")
        return jsonify({"success": False, "message": f"حدث خطأ داخلي: {str(e)}"}), 500

@app.route('/retrival')
def legal_qa():
    try:
        question = request.args.get("question", "")
        context = [x.page_content for x in retriever.invoke(question)]
        
        prompt = f"""
        أنت مستشار قانوني افتراضي. أجب بناء على النصوص التالية:
        {''.join(context)}
        
        ملاحظات:
        1. اجب باللغة العربية
        2. التزم بالنص القانوني فقط
        3. كن دقيقاً ومحايداً
        4. إذا لم تجد الإجابة قل 'لا أعرف'
        
        السؤال: {question}
        """
        
        # يمكن إضافة نموذج الذكاء الاصطناعي هنا
        answer = "هذا نموذج أولي - تحتاج لإضافة نموذج الذكاء الاصطناعي هنا"
        
        return jsonify({'question': question, 'answer': answer})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/chatbot1')
def general_qa():
    question = request.args.get("question", "")
    context = request.args.get("context", "")
    
    # يمكنك إضافة اتصال بنموذج خارجي هنا
    return jsonify({
        "reply": {
            "question": question,
            "context": context,
            "answer": "هذه إجابة نموذجية - تحتاج لتوصيل النموذج الفعلي"
        }
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
