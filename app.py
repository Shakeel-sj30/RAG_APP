import os
import streamlit as st

from rag_utility import process_document_to_chroma_db, answer_question

working_dir = os.path.dirname(os.path.realpath(__file__))

st.title("📄 Document RAG")

# session state to avoid reprocessing
if "processed" not in st.session_state:
    st.session_state.processed = False

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None and not st.session_state.processed:
    save_path = os.path.join(working_dir, uploaded_file.name)

    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    process_document_to_chroma_db(uploaded_file.name)

    st.session_state.processed = True
    st.success("✅ Document Processed Successfully")

user_question = st.text_area("Ask your question about the document")

if st.button("Answer"):
    if not user_question.strip():
        st.warning("⚠️ Please enter a question")
    elif not st.session_state.processed:
        st.warning("⚠️ Please upload and process a document first")
    else:
        answer = answer_question(user_question)

        st.markdown("### 🤖 Llama-3.3-70B Response")
        st.markdown(answer)