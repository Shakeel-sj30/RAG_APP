import os
from dotenv import load_dotenv

from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq

from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

working_dir = os.path.dirname(os.path.realpath(__file__))

# Embedding model
embeddings = HuggingFaceEmbeddings()

# LLM (Groq)
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)

def process_document_to_chroma_db(filename):
    loader = UnstructuredPDFLoader(f"{working_dir}/{filename}")
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200
    )

    texts = text_splitter.split_documents(documents)

    # ✅ No persist() needed (auto-save)
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