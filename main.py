from flask import Flask, jsonify, request, Response
import os
import requests as req
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
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


model_path = "./fine-tuned-xlm-roberta-law-model"

try:
    tokenizer = AutoTokenizer.from_pretrained(
        "lataon/xlm-roberta-base-finetuned-legal-domain",
        local_files_only=True,
        use_fast=True
    )
    
    model = AutoModelForQuestionAnswering.from_pretrained(
        "lataon/xlm-roberta-base-finetuned-legal-domain",
        local_files_only=True
    )
    
    # Verify embedding alignment
    if model.config.vocab_size != len(tokenizer):
        print("Resizing embeddings to match tokenizer...")
        model.resize_token_embeddings(len(tokenizer))
    
except Exception as e:
    print(f"Initialization error: {str(e)}")
    exit(1)
    
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

    # Get model predictions
    with torch.no_grad():
        outputs = model(**inputs)
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits

    # Find the start and end positions of the answer
    start_index = torch.argmax(start_logits)
    end_index = torch.argmax(end_logits)

    # Ensure the end index is greater than or equal to the start index
    if end_index < start_index:
        end_index = start_index

    # Decode the answer
    answer = tokenizer.decode(inputs["input_ids"][0][start_index:end_index + 1], skip_special_tokens=True)

    return jsonify({"reply": answer})

@app.route('/chatbot2')
def chatbot2():
    question = request.args["question"]
    answer="ok"
    return jsonify({'question': question, 'answer': answer})

if __name__ == '__main__':
    app.run(debug=True, port=os.getenv("PORT", default=5000))
