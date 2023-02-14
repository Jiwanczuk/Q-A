import streamlit as st
from utils import parse_file, embed_text, get_answer
from sidebar import sidebar

sidebar()

st.header("Doc QA")
uploaded_file = st.file_uploader("Upload a pdf or txt file", type=["pdf", "txt"])

if uploaded_file is not None:
    index = embed_text(parse_file(uploaded_file))
    query = st.text_area("Ask a question about the document")
    button = st.button("Submit")
    if button:
        st.write(get_answer(index, query))


