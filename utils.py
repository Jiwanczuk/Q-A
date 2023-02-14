from flask import Flask, request
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.vectorstores.faiss import FAISS
from ratelimiter import RateLimiter
import io

app = Flask(__name__)

def parse_file(file):
    file_extension = file.filename.split(".")[-1]
    if file_extension == "pdf":
        return parse_pdf(file)
    elif file_extension == "txt":
        return file.read().decode("utf-8")
    else:
        return None

def parse_pdf(file):
    pdf = PdfReader(file)
    output = []
    for page in pdf.pages:
        text = page.extract_text()
        output.append(text)
        

    return "\n\n".join(output)

def parse_txt(file):
    with open(file, 'r') as f:
        text = f.read()
        
        
    return "\n\n".join(output)

def embed_text(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=0, separators=["\n\n", ".", "?", "!", " ", ""]
    )
    texts = text_splitter.split_text(text)

    embeddings = OpenAIEmbeddings()
    index = FAISS.from_texts(texts, embeddings)

    return index

def get_answer(index, query):
    docs = index.similarity_search(query)

    chain = load_qa_chain(OpenAI(temperature=0))
    answer = chain.run(input_documents=docs, question=query)

    return answer

# Set the rate limit to 60 requests per minute
rate_limiter = RateLimiter(max_calls=60, period=60)

@app.route("/answer", methods=["POST"])
@rate_limiter
def answer():
    text = request.form.get("text")
    context = request.form.get("context")
    query = text

    # Use the context to augment the question
    if context:
        query = f"{context} {query}"

    index = embed_text(parse_file("file.pdf"))
    answer = get_answer(index, query)

    return answer








