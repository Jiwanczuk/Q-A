from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.vectorstores.faiss import FAISS
import streamlit as st


@st.cache
def parse_pdf(file):
    pdf = PdfReader(file)
    output = []
    for page in pdf.pages:
        text = page.extract_text()
        output.append(text)

    return "\n\n".join(output)


@st.cache
def parse_txt(file):
    with open(file, "r") as f:
        return f.read()


@st.cache
def embed_text(text):
    """Split the text and embed it in a FAISS vector store"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=0, separators=["\n\n", ".", "?", "!", " ", ""]
    )
    texts = text_splitter.split_text(text)

    embeddings = OpenAIEmbeddings()
    index = FAISS.from_texts(texts, embeddings)

    return index


def get_answer(index, query):
    """Returns answer to a query using langchain QA chain"""

    docs = index.similarity_search(query)

    chain = load_qa_chain(OpenAI(temperature=0))
    answer = chain.run(input_documents=docs, question=query)

    return answer


uploaded_file = st.file_uploader("Upload a document (PDF or TXT)", type=["pdf", "txt"])

if uploaded_file is not None:
    file_type = uploaded_file.split(".")[-1].lower()
    if file_type == "pdf":
        text = parse_pdf(uploaded_file)
    elif file_type == "txt":
        text = parse_txt(uploaded_file)
    else:
        st.error("Invalid file type")
        text = None

    if text:
        index = embed_text(text)
        query = st.text_area("Ask a question about the document")
        button = st.button("Submit")
        if button:
            st.write(get_answer(index, query))








