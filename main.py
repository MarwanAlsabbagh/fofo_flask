from flask import Flask, jsonify, request, Response
import os
import requests as req
# from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch
import warnings
# Correct local path format
from transformers.utils import logging

# Suppress PyTorch deprecation warnings
warnings.filterwarnings(
    "ignore", 
    message="torch.utils._pytree._register_pytree_node is deprecated"
)

# Set Transformers logging level
logging.set_verbosity_error()


from transformers import pipeline
pipe = pipeline(
    "question-answering",
    model="./fine-tuned-xlm-roberta-law-model",
    tokenizer="./fine-tuned-xlm-roberta-law-model"
)

app = Flask(__name__)

@app.route('/chatbot1')
def chatbot1():
    data = request.json
    question = data["question"]
    context = data["context"]

    # Tokenize the input (question + context)
    inputs = tokenizer(
        question,
        context,
        max_length=512,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )

   arabic_context = """
    المادة 12 من الدستور المصري تنص على أن التعليم حق لكل مواطن، هدفه بناء الشخصية المصرية، 
    الحفاظ على الهوية الوطنية، وتأكيد قيم المنهج العلمي، وتنمية المواهب، وتشجيع الابتكار.
    """
    
    arabic_question = "ماذا تنص المادة 12 من الدستور المصري عن التعليم؟"
    
    # 3. Get answer
    result = pipe(
        question=arabic_question,
        context=arabic_context,
        max_answer_len=100,
        handle_impossible_answer=False
    )
    
    print(f"الإجابة: {result['answer']}")
    print(f"الثقة: {result['score']:.2f}")

    return jsonify({"reply": result['answer']})

@app.route('/chatbot2')
def chatbot2():
    question = request.args["question"]
    answer="ok"
    return jsonify({'question': question, 'answer': answer})

if __name__ == '__main__':
    app.run(debug=True, port=os.getenv("PORT", default=5000))
