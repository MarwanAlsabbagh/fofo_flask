import re, requests, os, random, time, html
# from langchain.retrievers import BM25Retriever
from langchain_community.retrievers import BM25Retriever
# import google.generativeai as genai
from bs4 import BeautifulSoup
from urllib.parse import unquote
from langchain.text_splitter import RecursiveCharacterTextSplitter

from flask import Flask, jsonify, request, Response
import os
import requests
import torch
import warnings
from transformers.utils import logging
import requests


url = "https://groups.google.com/g/syrianlaw/c/Wba7S8LT9MU?pli=1"
urls =[]
response = requests.get(url)
if response.status_code == 200:
    soup = BeautifulSoup(response.content, "html.parser")
    a_tags = soup.find_all("a")
    for tag in a_tags:
        urls.append(tag.get("href"))
else:
    print(f"Failed to fetch {url}. Status code: {response.status_code}")

###################################

download_dir="/content/download_dir_rag"
os.makedirs(download_dir, exist_ok=True)
pages = []
def down(url):
    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            content_disposition = response.headers.get('Content-Disposition', '')
            filename = None
            if 'filename=' in content_disposition:
                filename = unquote(content_disposition.split('filename=')[1].split(';')[0].strip('"\''))
                filename = filename.encode('latin-1').decode('utf-8')
            else:
                filename = response.iter_content[0:100]+".txt"
            # if filename.find(".pdf") == -1:
            if len(str(filename))>90 and filename.find(".pdf") == -1:
                filename=str(str(filename[0:80])+".txt")
            
            file_path = os.path.join(download_dir, filename)
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=100):#10000
                    if chunk:
                        f.write(chunk)
            with open(file_path, 'rb') as tmp_file:
                raw_content = tmp_file.read()
                try:
                    content = raw_content.decode('utf-8')
                except UnicodeDecodeError:
                    content = raw_content.decode('latin-1', errors='replace')
                content = content.replace("‏","").replace("بشار الأسد","").replace('\n\n', ' ').replace('\r\n', ' ').replace('\u200c', '')
            with open(file_path, 'wb') as tmp_file:
                tmp_file.write(content.encode('utf-8'))
            pages.append({
            'content': content,
            'title': filename,
            })
            # print(f"Successfully downloaded: {filename}")
        else:
            print(f"Failed to download. Status code: {response.status_code}")
    except Exception as e:
        print(f"An error occurred: {e}")

# to download the files
for url in urls[::5]:
    if url.find("https://docs") != -1:
        down(url)
print('All docs installed') 

##############################################################

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 80058,#حجم كل جزء (مقطع) بعد التقسيم بالعدد الإجمالي للأحرف.
    chunk_overlap  = 5000,#عدد الأحرف المشتركة بين كل جزء والجزء الذي يليه (لضمان عدم فقدان المعلومات عند التجزئة).
    is_separator_regex = True,#يحدد ما إذا كان الفاصل المستخدم لفصل الأجزاء عبارة عن تعبير منتظم (regex) أم مجرد نص عادي (False يعني نص عادي).
    separators=["\ufeff"]
)

final_pages=[]
for i in pages:
    if len(i['content'])<=6500:
        final_pages.append(i)

final_pages[0]['content'].replace('\ufeff', '')
all_pages = [elm['content'] for elm in final_pages]
texts = text_splitter.create_documents(all_pages)
chunks=[elm for elm in texts]
docs = [doc.page_content.replace('\ufeff', '') for doc in chunks]
print(f"pages: {len(pages)}")
print(f"final_pages: {len(final_pages)}")
print(f"chanks: {len(chunks)}")

##############################################################

retriever = BM25Retriever.from_texts(texts=docs,ngram_range=(2, 2),k=3)

##############################################################
# genai.configure(api_key='AIzaSyCnaJnmBKGH-KLMzAqSqqTFcUnuQpCNatc')
models_params = {
    "temperature": 0.4, # 1
    "top_p": 0.95,
    # "top_k": 40,
    "max_output_tokens": 1800
}

# best_model = genai.GenerativeModel(model_name="tunedModels/parliamenttunedmodel4-bkk8kcf0901g",generation_config=models_params)

# chat_session = best_model.start_chat(history=[])

def tune_question_answering(user_question):

    context = [ x.page_content for x in retriever.invoke(user_question)]
    template = f"""ﺃﻧﺖ ﻣﺴﺘﺸﺎﺭ ﻗﺎﻧﻮﻧﻲ ﺍﻓﺘﺮﺍﺿﻲ. ﺳﺘﺠﻴﺐ ﻋﻠﻰ ﺍﻷﺳﺌﻠﺔ ﺍﻟﻘﺎﻧﻮﻧﻴﺔ ﺑﻨﺎﺀ ﻋﻠﻰ ﺍﻟﻨﺼﻮﺹ ﺍﻟﻘﺎﻧﻮﻧﻴﺔ ﺍﻟﻤﻘﺪﻣﺔ.

       ﺍﻟﻨﺺ ﺍﻟﻘﺎﻧﻮﻧﻲ:
       {''.join(context)}
       ﻳﺮﺟﻰ ﻃﺮﺡ ﺍﻷﺳﺌﻠﺔ، ﻭﺳﺘﻘﺪﻡ ﺍﻹﺟﺎﺑﺎﺕ ﺍﻟﻘﺎﻧﻮﻧﻴﺔ ﺍﻟﺪﻗﻴﻘﺔ ﺑﻨﺎﺀ ﻋﻠﻰ ﺍﻟﻨﺼﻮﺹ ﺍﻟﻤﺘﺎﺣﺔ.
       ﺑﻌﺾ ﺍﻟﻤﻼﺣﻈﺎﺕ ﺍﻟﺬﻱ ﻳﺠﺐ ﺍﺗﺒﺎﻋﻬﺎ
       1. ﺍﻻﺟﺎﺑﺔ ﻳﺠﺐ ﺍﻥ ﺗﻜﻮﻥ ﺑﺎﻟﻠﻐﺔ ﺍﻟﻌﺮﺑﻴﺔ
       2. ﺍﻻﺟﺎﺑﺔ ﻳﺠﺐ ﺍﻥ ﺗﻜﻮﻥ ﻣﻘﺘﺼﺮﺓ ﻋﻠﻰ ﺍﻟﻨﺺ ﺍﻟﻘﺎﻧﻮﻧﻲ ﺍﻟﻤﻘﺪﻡ ﻓﻘﻂ
       3. ﻳﺠﺐ ﺍﻥ ﺗﻜﻮﻥ ﺍﻻﺟﺎﺑﺔ ﺩﻗﻴﻘﺔ ﻭﻣﺤﺎﻳﺪﺓ
       4. ﻓﻲ ﺣﺎﻝ ﺍﻧﻚ ﻻ ﺗﻌﺮﻑ ﺍﻻﺟﺎﺑﺔ ﻓﻘﻂ ﺍﺟﺐ ﺑﺎﻧﻚ ﻻ ﺗﻌﺮﻑ ﺍﻟﺠﻮﺍﺏ
    
     سؤال: {user_question}"""
    # answer = chat_session.send_message(template).text
    answer = 'asd'

    print('Question:')
    print(user_question)
    print("------------------------------------------------")
    print('answer:')
    print(answer)
    return answer

##############################################################

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()
            
app = Flask(__name__)

@app.route('/chatbot')
def chatbot():
    try:
        question = request.args["question"].replace("%20"," ")

        token = request.args["question"]

        API_URL = "https://router.huggingface.co/hyperbolic/v1/chat/completions"
        
        headers = {
            "Authorization": "Bearer " + token,
        }

        # 2. Define input properly
        # context = request.args["context"].replace("%20"," ")
        # المادة 12 من الدستور المصري تنص على أن التعليم حق لكل مواطن، هدفه بناء الشخصية المصرية، الحفاظ على الهوية الوطنية، وتأكيد قيم المنهج العلمي، وتنمية المواهب، وتشجيع الابتكار

        response = query({
            "messages": [
                {
                    "role": "user",
                    "content": question
                }
            ],
            "max_tokens": 512,
            "model": "deepseek-ai/DeepSeek-V3-0324"
        })
        
        output = response["choices"][0]["message"]['content']
        return jsonify({"reply": output})
    except Exception as e:
        print(e)
    return jsonify({"reply": "error"})

@app.route('/retrival')
def retrival():
    try:
        question = request.args["question"]
        answer=tune_question_answering(question)
        return jsonify({'question': question, 'answer': answer})
    except:
         return jsonify({'question': question, 'answer': "error server"})

if __name__ == '__main__':
    app.run(debug=True, port=os.getenv("PORT", default=5000))
