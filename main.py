from flask import Flask, jsonify, request, Response
import os
import requests
import torch
import warnings
from transformers.utils import logging
warnings.filterwarnings(
    "ignore", 
    message="torch.utils._pytree._register_pytree_node is deprecated"
)
logging.set_verbosity_error()


app = Flask(__name__)

@app.route('/chatbot1')
def chatbot1():
    try:
        question = request.args["question"].replace("%20"," ")
        # context =  request.args["context"]
    
        # 2. Define input properly
        arabic_context = """
        المادة 12 من الدستور المصري تنص على أن التعليم حق لكل مواطن، هدفه بناء الشخصية المصرية، 
        الحفاظ على الهوية الوطنية، وتأكيد قيم المنهج العلمي، وتنمية المواهب، وتشجيع الابتكار.
        """

        API_URL = "https://router.huggingface.co/hf-inference/v1"
        headers = {"Authorization": "Bearer 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"}
        
        def query(payload):
        	response = requests.post(API_URL, headers=headers, json=payload)
        	return response.json()
        	
        output = query({
        	"inputs": {
        	"question": "What is my name?",
        	"context": "My name is Clara and I live in Berkeley."
        },
        })
        print(output)
        return jsonify({"reply": output})
    except Exception as e:
        print(e)
        return jsonify({"reply": "error"})

@app.route('/chatbot2')
def chatbot2():
    question = request.args["question"]
    answer="ok"
    return jsonify({'question': question, 'answer': answer})

if __name__ == '__main__':
    app.run(debug=True, port=os.getenv("PORT", default=5000))
