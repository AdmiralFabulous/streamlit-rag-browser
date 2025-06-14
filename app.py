import os, tempfile, pathlib
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA

st.set_page_config(page_title="Browser‑RAG", page_icon="📚", layout="wide")
st.title("📚 Chat with Your Documents (Browser‑Only)")

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.sidebar.header("Upload documents")
uploaded_files = st.sidebar.file_uploader(
    "Drag & drop or select files",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True
)

if uploaded_files and st.sidebar.button("Process files"):
    docs = []
    for file in uploaded_files:
        suffix = pathlib.Path(file.name).suffix.lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name

        if suffix == ".pdf":
            loader = PyPDFLoader(tmp_path)
        elif suffix == ".docx":
            loader = Docx2txtLoader(tmp_path)
        else:
            loader = TextLoader(tmp_path, encoding="utf-8")

        docs.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024, chunk_overlap=128
    )
    chunks = text_splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vectordb = Chroma.from_documents(
        chunks, embeddings, collection_name="user_docs"
    )

    st.session_state.vectorstore = vectordb
    st.success(f"Processed {len(uploaded_files)} file(s) and {len(chunks)} chunks.")

question = st.chat_input("Ask me anything about your documents…")

if question:
    if st.session_state.vectorstore is None:
        st.error("Please upload and process documents first.")
    else:
        retriever = st.session_state.vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": 6}
        )
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )

        result = qa_chain({"query": question})
        answer = result["result"]
        sources = result["source_documents"]

        st.session_state.chat_history.append(("user", question))
        st.session_state.chat_history.append(("assistant", answer))

        for role, msg in st.session_state.chat_history:
            st.chat_message(role).write(msg)

        with st.expander("Show sources"):
            for doc in sources:
                st.write(f"**{doc.metadata.get('source', 'Unknown')}**")
                st.write(doc.page_content[:400] + "…")
