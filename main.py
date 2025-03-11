from flask import Flask, jsonify, request, Response
import os
import requests as req
# from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from transformers import pipeline
import torch
import warnings
from transformers.utils import logging
warnings.filterwarnings(
    "ignore", 
    message="torch.utils._pytree._register_pytree_node is deprecated"
)
logging.set_verbosity_error()

pipe = pipeline(
    "question-answering",
    model="deepset/xlm-roberta-large-squad2",
    tokenizer="deepset/xlm-roberta-large-squad2",
    use_auth_token="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",
    device_map="auto",
    max_memory={0: "5GB"}
)

app = Flask(__name__)

@app.route('/chatbot1')
def chatbot1():
    try:
        question = request.args["question"]
        # context =  request.args["context"]
    
        # 2. Define input properly
        arabic_context = """
        المادة 12 من الدستور المصري تنص على أن التعليم حق لكل مواطن، هدفه بناء الشخصية المصرية، 
        الحفاظ على الهوية الوطنية، وتأكيد قيم المنهج العلمي، وتنمية المواهب، وتشجيع الابتكار.
        """
        
        # arabic_question = "ماذا تنص المادة 12 من الدستور المصري عن التعليم؟"
        
        # 3. Get answer
        result = pipe(
            question=question,
            context=arabic_context,
            max_answer_len=100,
            handle_impossible_answer=False
        )
        
        print(f"الإجابة: {result['answer']}")
        print(f"الثقة: {result['score']:.2f}")
    except Exception as e:
        return jsonify({"Exception": e})
    return jsonify({"reply": result['answer']})

@app.route('/chatbot2')
def chatbot2():
    question = request.args["question"]
    answer="ok"
    return jsonify({'question': question, 'answer': answer})

if __name__ == '__main__':
    app.run(debug=True, port=os.getenv("PORT", default=5000))
