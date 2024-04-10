from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from tqdm.auto import tqdm
from dotenv import load_dotenv

load_dotenv()
os.getenv("GOOGLE_API_KEY")


def text_formatter(text: str) -> str:
    cleaned_text = text.replace("\n", " ").strip() 

    return cleaned_text


def get_pdf_text(pdf_docs):
    raw_text = ""
    for dirname, _, filenames in os.walk(pdf_docs):
        for filename in filenames:
            pdf_path = str(os.path.join(dirname, filename))
            doc = PdfReader(pdf_path)
            for page in doc.pages:
                raw_text += page.extract_text()
    return raw_text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def main():
    pdf_docs = "data"
    raw_text = get_pdf_text(pdf_docs)
    text_chunks = get_text_chunks(raw_text)
    get_vector_store(text_chunks)
    print("Done")

main()