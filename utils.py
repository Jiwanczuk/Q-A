import io
import fitz
import re
from ratelimiter import RateLimiter
from langdetect import detect
from summarizer import summarize

@RateLimiter(max_calls=1, period=1)
def get_answer(index, query):
    query = re.sub(r"[^\w\s]", "", query).lower()
    answer = index.search(query)
    if answer:
        return answer[0][1]
    else:
        return "Sorry, I couldn't find an answer to that."

def parse_file(file):
    file_extension = file.filename.split(".")[-1]
    if file_extension == "pdf":
        return parse_pdf(file)
    elif file_extension == "txt":
        return parse_text(file)

def parse_pdf(file):
    pdf_bytes = file.getvalue()
    pdf = fitz.open(io.BytesIO(pdf_bytes))
    output = []
    for page in pdf:
        output.append(page.get_text("text").strip())
    text = "\n\n".join(output)
    return text

def parse_text(file):
    text = file.read().decode("utf-8")
    return text

def embed_text(text):
    lang = detect(text)
    if lang == "en":
        return summarize(text, ratio=0.2)
    else:
        return text









