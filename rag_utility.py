import os
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq

from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# ✅ Use cloud-safe temp directory
working_dir = "/tmp"

# ✅ Stable embedding model (important for cloud)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ✅ FIX: Use Streamlit secrets
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    groq_api_key=st.secrets["GROQ_API_KEY"]
)


def process_document_to_chroma_db(filename):
    loader = PyPDFLoader(f"{working_dir}/{filename}")
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200
    )

    texts = text_splitter.split_documents(documents)

    vectordb = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory=f"{working_dir}/doc_vectorstore"
    )

    return vectordb


def answer_question(user_question):
    vectordb = Chroma(
        persist_directory=f"{working_dir}/doc_vectorstore",
        embedding_function=embeddings
    )

    retriever = vectordb.as_retriever()

    prompt = ChatPromptTemplate.from_template(
        """Answer the question based only on the context below:

        {context}

        Question: {question}
        """
    )

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain.invoke(user_question)
