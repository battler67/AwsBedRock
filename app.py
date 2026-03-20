import json,os,sys,boto3
from langchain_aws import BedrockEmbeddings
from langchain_aws import ChatBedrock
from langchain_community.llms.bedrock import Bedrock
from langchain_text_splitters  import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
import numpy as np
import streamlit as st 
bedrock_client = boto3.client(service_name = "bedrock-runtime")
bedrockEmbeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0",client=bedrock_client)

def data_ingestion():
    loader = PyPDFLoader("data/ML_BIBLE.pdf")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000,chunk_overlap=1000)
    if(0):
        docs = text_splitter.split_documents(documents=documents)
    vector_Store_faiss = FAISS.from_documents(
        documents=docs,embedding=bedrockEmbeddings
    )
    vector_Store_faiss.save_local("faiss_index")
    faiss_index = FAISS.load_local(
        "faiss_index",
        bedrockEmbeddings,
        allow_dangerous_deserialization=True
    )
    docs = list(faiss_index.docstore._dict.values())
    return docs

def get_calude_llm():
    llm = ChatBedrock(model="anthropic.claude-3-5-sonnet-20240620-v1:0",
        region_name="us-east-1",
        max_tokens=1024,           # Max response length
        temperature=0.5,           # How creative (0=deterministic, 1=random)
        top_p=0.9,                 # Nucleus sampling
        top_k=50 
    )
    return llm

def get_llama_llm():
    llm = ChatBedrock(model_id="meta.llama3-70b-instruct-v1:0",client=bedrock_client,model_kwargs={"max_gen_len":512})
    return llm

prompt_template = """
    This is the context {context} and this is the input question {question} ALso provide me text in {context}
"""
PROMT = PromptTemplate(template=prompt_template,input_variables=["context","question"])

def get_response_llm(llm,vectoreStore_Faiss,query):
    qa =  RetrievalQA.from_chain_type(llm=llm,chain_type="stuff",
                                    retriever = vectoreStore_Faiss.as_retriever(search_type="similarity",search_kwargs={"k":3}),
                                    return_source_documents = True,chain_type_kwargs = {"prompt":PROMT}
                                    )
    answer = qa({"query":query})
    return answer['result']

def  main():
    st.set_page_config("Chat Pdf")
    st.header("Chat with PDF using AWS Bedrock💁")
    user_question = st.text_input(label="enter ur question")
    with st.sidebar:
        st.title("Update or create vector store:")
        if st.button("Vectors Update"):
            with st.spinner("Processing..."):
                data_ingestion()
                st.success("Done")
        
    if st.button("Claude Output"):
        with st.spinner("Processing..."):
            faiss_index = FAISS.load_local("faiss_index",bedrockEmbeddings,allow_dangerous_deserialization=True)
            llm  = get_calude_llm()
            st.write(get_response_llm(llm,vectoreStore_Faiss=faiss_index,query=user_question))
            st.success("Done")
    
    if st.button("llama Output"):
        with st.spinner("Processing..."):
            faiss_index = FAISS.load_local("faiss_index",bedrockEmbeddings,allow_dangerous_deserialization=True)
            llm  = get_llama_llm()
            st.write(get_response_llm(llm,vectoreStore_Faiss=faiss_index,query=user_question))
            st.success("Done")

if __name__ == "__main__":
    main()