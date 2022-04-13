from haystack.question_generator import QuestionGenerator
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
from pydantic import BaseModel
import uvicorn
from haystack.file_converter import ImageToTextConverter
from haystack.document_stores import ElasticsearchDocumentStore
from haystack.nodes import EmbeddingRetriever
import pandas as pd
import os
from haystack.pipelines import FAQPipeline
from haystack.nodes import ElasticsearchRetriever
from haystack.utils import clean_wiki_text, convert_files_to_dicts
from haystack.nodes import FARMReader
from haystack.pipelines import ExtractiveQAPipeline
import s3fs
import boto3
import fitz

s3 = boto3.client('s3', aws_access_key_id='AKIAUOJDVU6XYMY4EPMZ' , aws_secret_access_key='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')

converter = ImageToTextConverter(remove_numeric_tables=True, valid_languages=["eng"])

document_store = ElasticsearchDocumentStore(host="localhost", username="", password="",
                                            index="document",
                                            embedding_field="question_emb",
                                            embedding_dim=384,
                                            excluded_meta_data=["question_emb"])

retrieverCSV = EmbeddingRetriever(document_store=document_store, embedding_model="sentence-transformers/all-MiniLM-L6-v2", use_gpu=True)

retriever = ElasticsearchRetriever(document_store=document_store)
reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=True)

pipedocs = ExtractiveQAPipeline(reader, retriever)


pipe = FAQPipeline(retriever=retrieverCSV)

qg = QuestionGenerator()

out_file_path = ""

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

class Item(BaseModel):
    paragraph: List[str] = []

@app.get('/')
async def root():
    return {'hello': 'world'}

@app.post("/items/")
async def QuestionGenerator(customerName: str, filename: str, inputfilebucket: str, outputquestionbucket: str):
    try: 
        os.makedirs("s3_downloads", exist_ok=True)
        os.makedirs("s3_downloads/" + customerName, exist_ok=True)
        file_location = os.getcwd() + "/s3_downloads/" + customerName + "/" + filename
        s3.download_file(inputfilebucket, customerName + "/" + filename, 's3_downloads/' + customerName + "/" + filename)
        if filename.endswith(".pdf"):
            with fitz.open(file_location) as doc:
                text = ""
                for page in doc:
                    text += page.getText()
            print(text)
            if os.path.exists(file_location):
                os.remove(file_location)
            else:
                print("The file does not exist")
            file_location = os.getcwd() + "/s3_downloads/" + customerName + "/" + filename + ".txt"
            file1 = open(file_location,"w")
            file1.writelines(text)
            file1.close()
        dicts = convert_files_to_dicts(dir_path="s3_downloads/" + customerName + '/', clean_func=clean_wiki_text, split_paragraphs=True)
        dicts[0]['customerName'] = customerName
        dicts[0]['type'] = 'txt'
        dicts[0]['name'] = filename
        document_store.write_documents(dicts)
        filter = {"name": [filename], "customerName" : [customerName]}
        document_store.update_embeddings(retrieverCSV,filters=filter)
        result = ""
        with open(file_location) as f:
            text = f.read()
            result = qg.generate(text)
        if os.path.exists(file_location):
            os.remove(file_location)
        else:
            print("The file does not exist")
        list_dataframe = pd.DataFrame(result)
        bytes_to_write = list_dataframe.to_csv(None).encode()
        fs = s3fs.S3FileSystem(anon=False, key='AKIAUOJDVU6XYMY4EPMZ', secret='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')
        with fs.open('s3://'+ outputquestionbucket +'/'+ customerName + '/' + filename + '.csv', 'wb') as f:
            f.write(bytes_to_write)
        return list_dataframe
    except Exception as e:
        return {'error': e}

@app.post("/upload-file/")
async def ImageToText(customerName: str, filename: str, inputfilebucket: str, outputquestionbucket: str):
    try:
        os.makedirs("image_downloads", exist_ok=True)
        os.makedirs("image_downloads/" + customerName, exist_ok=True)
        file_location = os.getcwd() + "/image_downloads/" + customerName + "/" + filename
        s3.download_file(inputfilebucket, customerName + "/" + filename, 'image_downloads/' + customerName + "/" + filename)
        doc = converter.convert(file_path=file_location, meta=None)
        if os.path.exists(file_location):
            os.remove(file_location)
        else:
            print("The file does not exist")
        # print(doc)
        os.makedirs("images", exist_ok=True)
        os.makedirs("images/" + customerName, exist_ok=True)
        contents = doc['content']
        print(contents)
        file_location = os.getcwd() + "/images/" + customerName + "/" + filename + ".txt"
        with open(file_location, "wb+") as file_object:
            file_object.write(contents.encode())
        fs = s3fs.S3FileSystem(anon=False, key='AKIAUOJDVU6XYMY4EPMZ', secret='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')
        with fs.open('s3://'+ inputfilebucket +'/'+ customerName + '/' + filename + '.txt', 'wb') as f:
            f.write(contents.encode())
        dicts = convert_files_to_dicts(dir_path=("images/" + customerName + "/"))
        dicts[0]['customerName'] = customerName
        dicts[0]['name'] = filename
        dicts[0]['type'] = 'image'
        # print(dicts)
        document_store.write_documents(dicts)
        filter = {"name": [filename], "customerName" : [customerName]}
        document_store.update_embeddings(retrieverCSV,filters=filter)
        generatequestion(outputquestionbucket,customerName,filename)
        if os.path.exists(file_location):
            os.remove(file_location)
        else:
            print("The file does not exist")
        return {"info": f"file '{filename}' saved at '{file_location}'",
                "data": doc}
    except Exception as e:
        return {'error': e}

def generatequestion(outputquestionbucket,customerName,filename):
    file_location = os.getcwd() + "/images/" + customerName + "/" + filename + ".txt"
    result = ""
    with open(file_location) as f:
        text = f.read()
        result = qg.generate(text)
    if os.path.exists(file_location):
        os.remove(file_location)
    else:
        print("The file does not exist")
    list_dataframe = pd.DataFrame(result)
    bytes_to_write = list_dataframe.to_csv(None).encode()
    fs = s3fs.S3FileSystem(anon=False, key='AKIAUOJDVU6XYMY4EPMZ', secret='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')
    with fs.open('s3://'+ outputquestionbucket +'/'+ customerName + '/' + filename + '.csv', 'wb') as f:
        f.write(bytes_to_write)

@app.post("/csv-faq-file/")
async def CSVFAQ(customerName: str, filename: str, inputfilebucket: str, outputquestionbucket: str):
    try:
        os.makedirs("csv_downloads", exist_ok=True)
        os.makedirs("csv_downloads/" + customerName, exist_ok=True)
        file_location = os.getcwd() + "/csv_downloads/" + customerName + "/" + filename
        s3.download_file(inputfilebucket, customerName + "/" + filename, 'csv_downloads/' + customerName + "/" + filename)
        df = pd.read_csv(file_location)
        df.fillna(value="", inplace=True)
        df["question"] = df["question"].apply(lambda x: x.strip())
        df['filename'] = filename
        df['customerName'] = customerName
        df['type'] = 'csv'
        print(df.head())
        questions = list(df["question"].values)
        df["question_emb"] = retrieverCSV.embed_queries(texts=questions)
        df = df.rename(columns={"question": "content"})
        docs_to_index = df.to_dict(orient="records")
        # print(docs_to_index)
        if os.path.exists(file_location):
            os.remove(file_location)
        else:
            print("The file does not exist")
        document_store.write_documents(docs_to_index)
        return {"info": f"file '{file_location}' saved at '{file_location}'",
                "status": "file saved successfully"}
    except Exception as e:
        return {'error': e}


@app.post("/csv-query/")
async def FAQQuestion(text: str, customerName: str, top_k: Optional[int] = 1):
    try:
        filter = {"customerName": [customerName]}
        prediction = pipe.run(query=text, params={"Retriever": {"top_k": top_k},"filters":filter})
        return prediction
    except Exception as e:
        return {'error': e}


@app.post("/docs-query/")
async def DOCSQuestion(text: str, customerName: str, top_k: Optional[int] = 1):
    try:
        filter = {"customerName": [customerName]}
        prediction = pipedocs.run(query=text, params={"Retriever": {"top_k": top_k},"filters":filter})
        return prediction
    except Exception as e:
        return {'error': e}

@app.post("/delete-csv/")
async def DeleteCSV(filename: str, customerName: str):
    try:
        filter = {"filename": [filename], "customerName" : [customerName]}
        document_store.delete_all_documents("document", filter)
        return "documents deleted successfully"
    except Exception as e:
        return {'error': e}

@app.post("/delete-docs/")
async def DeleteDocs(text: str, customerName: str):
    try:
        filter = {"name": [text], "customerName" : [customerName]}
        document_store.delete_all_documents("document", filter)
        return "documents deleted successfully"
    except Exception as e:
        return {'error': e}

@app.post("/delete-all/")
async def DeleteAll():
    try:
        document_store.delete_all_documents("document")
        return "All documents deleted successfully"
    except Exception as e:
        return {'error': e}

#pypy3 -m pip install --extra-index https://antocuni.github.io/pypy-wheels/ubuntu cpython numpy

uvicorn.run(app, host="0.0.0.0", port=8001)

#sudo docker run -d -p 9200:9200 -e "discovery.type=single-node" elasticsearch:7.9.2

