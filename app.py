import streamlit as st
import openai
import validators

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import UnstructuredURLLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.vectorstores import FAISS
import os

## Load the environment variables
from dotenv import load_dotenv
load_dotenv()

## Langsmith Tracking
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_PROJECT"]="Simple Q&A Chatbot With OPENAI"

## Set up the streamlit APP
st.set_page_config(page_title="Veeva documentation chatbot", page_icon="🦜")
st.title("🦜 Veeva documentation chatbot")

## Get the OpenAI API Key, select model and  website URL to be question
with st.sidebar:
    ## Sidebar for settings
    st.sidebar.title("Settings")
    ## OpenAI API Key
    openai_api_key=st.text_input("OpenAI API Token",value="",type="password")
    ## Select the OpenAI model
    model_engine=st.selectbox("Select Open AI model",["gpt-4o","gpt-4-turbo","gpt-4"])
    ## Adjust response parameter
    temperature=st.slider("Temperature",min_value=0.0,max_value=1.0,value=0.7)
    #max_tokens =st.slider("Max Tokens", min_value=50, max_value=300, value=150)
    ## MAin interface for user input

st.write("Goe ahead and ask any question")
user_input=st.text_input("Ask a question:")
st.write("")

## Capture web URL
st.write("Enter the URL of the Veeva documentation page")
generic_url=st.text_input("URL",label_visibility="collapsed")

os.environ['HF_TOKEN']=os.getenv("HF_TOKEN")
#embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
llm=ChatOpenAI(model=model_engine)

try:
    def create_chunk_retriever(docs ,fileName, ensemble=False):
        text_splitter_chunks=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
        splits=text_splitter_chunks.split_documents(docs)
        
        vectorstore=FAISS.from_documents(documents=splits,embedding=embeddings)
        #vectorstore=Chroma.from_documents(documents=splits,embedding=embeddings)
        vectorstore_retreiver=vectorstore.as_retriever()
        
        if ensemble:
            #HybrKeyword search
            keyword_retriever = BM25Retriever.from_documents(splits)
            keyword_retriever.k =  3
            #Hybrid search
            ensemble_retriever = EnsembleRetriever(retrievers=[vectorstore_retreiver,keyword_retriever],weights=[0.3, 0.7])
            return ensemble_retriever
        return vectorstore_retreiver
    
except Exception as e:
        st.exception(f"Chroma Exception:{e}")

def create_prompt():
## Prompt Template
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know."
        "\n\n"
        "{context}"
    )
    #Use three sentences maximum and keep the "answer concise.
    print("User input: ",user_input)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )
    return prompt

session_id=st.text_input("Session ID",value="default_session")
    ## statefully manage chat history

if 'store' not in st.session_state:
    st.session_state.store={}

if st.button("Ask from Veeva online documentation"):
    ## Validate all the inputs
    if not openai_api_key.strip() or not generic_url.strip():
        st.error("Please provide the information to get started")
    elif not validators.url(generic_url):
        st.error("Please enter a valid Url.")
    elif not "veeva" in generic_url:
        st.error("Please enter a valid Veeva Url.")

    else:
        try:
            with st.spinner("Waiting..."):

                ## loading the website data
                print(generic_url.split('/')[-1])
                fileName= generic_url.split('/')[-1]
                loader=UnstructuredURLLoader(urls=[generic_url],ssl_verify=False,
                                                 headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"})
                docs=loader.load()
                print("document loaded")
                ## Create a retriever
                retreiver = create_chunk_retriever(docs, fileName, True)
                print("retriever created")
                ## Prompt Template
                prompt = create_prompt()
                print("prompt created")

                ## RAG
                question_answer_chain=create_stuff_documents_chain(llm,prompt)
                print("chain created")
                rag_chain=create_retrieval_chain(retreiver,question_answer_chain)
                print("rag chain created")
                response=rag_chain.invoke({"input":user_input})
                print("response generated")
                st.success(response)

                if user_input:
                    #response=generate_response(user_input,openai_api_key,model_engine,temperature,max_tokens)
                    st.write(response['answer'])

                elif user_input:
                    st.warning("Please enter the OPen AI aPi Key in the sider bar")
                else:
                    st.write("Please provide the user input")
        except Exception as e:
            st.exception(f"Exception:{e}")
