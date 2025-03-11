from flask import Flask, jsonify, request, Response
import os
import requests as req
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch
# Correct local path format

model = AutoModelForQuestionAnswering.from_pretrained("/app/fine-tuned-xlm-roberta-law-model")
# For models on HuggingFace Hu
tokenizer = AutoTokenizer.from_pretrained("/app/fine-tuned-xlm-roberta-law-model")

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
