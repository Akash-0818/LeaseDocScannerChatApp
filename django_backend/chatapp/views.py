from django.shortcuts import render
from django.http import JsonResponse
import os
from django.views.decorators.csrf import csrf_protect
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import PyPDFLoader
#from langchain.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
import time
import warnings

warnings.filterwarnings("ignore")

load_dotenv()

chat_history = []

def home(request):
    return render(request, 'home.html')

def generate_response(question):
    embeddings = OpenAIEmbeddings()
    vectorstore = PineconeVectorStore(
            index_name=os.environ["INDEX_NAME"], embedding=embeddings
    )

    chat = ChatOpenAI(verbose=True, temperature=0, model_name="gpt-3.5-turbo")

    qa = ConversationalRetrievalChain.from_llm(
        llm=chat, chain_type="stuff", retriever=vectorstore.as_retriever()
    )  

    res = qa({"question": question, "chat_history": chat_history})

    history = (res["question"], res["answer"])
    chat_history.append(history)

    return res

@csrf_protect
def process_message(request):
    if request.method == "POST":
        print("post request")
        qn = request.POST.get('message','')
        response = generate_response(qn)
        return JsonResponse({'status': 'OK','response': response['answer']})
    
def run_background_task(request):
    print("Running background task...")
    
    #time.sleep(3)  # Simulate a long-running task
    loader = PyPDFLoader(r"C:\Users\akash\storage\Skye_CompletedLease.pdf")
    document = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(document)
    print(f"created {len(texts)} chunks")
    embeddings = OpenAIEmbeddings(openai_api_type=os.environ.get("OPENAI_API_KEY"))
    PineconeVectorStore.from_documents(texts, embeddings, index_name=os.environ.get("INDEX_NAME"))

    print("Background task completed!")
    return JsonResponse({"message": "Task finished"})