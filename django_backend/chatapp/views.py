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