import os
import pickle
import time
import streamlit as st

from langchain.llms.openai import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS

from flask import Flask, request, jsonify
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

@app.route('/newsScraperAI', methods=['POST'])
def news_scraper_ai():
    try:
        request_data = request.get_json()
        url=[];
        if(request_data and 'source_url' in request_data):
            url.append(request_data['source_url']);
        if(len(url) > 0):
            
            llm = OpenAI(temperature=0.9, max_tokens=500)
            embadding = OpenAIEmbeddings();
            filePath = 'faiss_store.pkl';
            loader = UnstructuredURLLoader(urls=url);
            data = loader.load();
            if(data and len(data) > 0):
                text_splitter = RecursiveCharacterTextSplitter(
                    separators=['\n\n', '\n', '.', ','],
                    chunk_size=1000
                )
                docs = text_splitter.split_documents(data)
                
                vectorStore_openai = FAISS.from_documents(docs,embadding);
                
                time.sleep(2);
                
                with open(filePath,"wb") as f:
                    pickle.dump(vectorStore_openai,f);
                    
                if(request_data and 'title' in request_data):
                    query = request_data['title'];
                    if query:
                        query = "make something interesting and insightful and exciting for humans and in 60 words from:" + query;
                        if os.path.exists(filePath):
                            with open(filePath, "rb") as f:
                                vectorstore = pickle.load(f)
                                chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
                                result = chain({"question": query}, return_only_outputs=True)
                                print("result=============================>>>>>>>>>>>>")
                                print(result);
                                response_data = {"message": "Data received successfully", "data": result}
                                return jsonify(response_data)
        else:
            response_data = {"message": "No URLs found", "data": request_data}

            # Return a JSON response
            return jsonify(response_data)
            
        
    except Exception as e:
        print(f"Error: {e}")
        # Handle exceptions if necessary
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
