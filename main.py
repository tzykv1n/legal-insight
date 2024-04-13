import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_together import Together
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
# from transformers import AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv

load_dotenv()
os.getenv("GOOGLE_API_KEY")
os.environ['TOGETHER_AI'] = "dd8a860b96bb5483c357fa72c6f5a1a41657cfefa2c0bbe578ed616a3facdddb"
TOGETHER_AI_API= os.environ['TOGETHER_AI']


def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_text_chunks2(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=500)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index_input")


def get_chain():

    prompt_template = """<s>[INST]This is a chat template and As a legal chat bot, your primary objective is to provide accurate and concise information based on the user's questions. Do not generate your own questions and answers. You will adhere strictly to the instructions provided, offering relevant context from the knowledge base while avoiding unnecessary details. Your responses will be brief, to the point, and in compliance with the established format. If the answer is not in provided context, you will rely on your own knowledge base to generate an appropriate response.  You will prioritize the user's query and refrain from posing additional questions. The aim is to deliver professional, precise, and contextually relevant information pertaining to the constitution of India or Indian Penal Code or Contract act of India. Try to include what all are the Indian Penal Code Sections involved in the user's question if required and only if it is related to Indian penal code. The answer should not contain "Based on context provided", "based on the above context", etc. No need to ask any extra questions back to the user.
    Context: {context}
    Question: {question}
    Answer:
    </s>[INST]
    """

    model = Together(
        model="mistralai/Mistral-7B-Instruct-v0.2",
        temperature=0.5,
        max_tokens=1024,
        together_api_key=f"{TOGETHER_AI_API}"
        )

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    # embeddings = HuggingFaceEmbeddings(model_name="nomic-ai/nomic-embed-text-v1",model_kwargs={"trust_remote_code":True,"revision":"289f532e14dbbbd5a04753fa58739e9ba766f3c7"})
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_chain()

    
    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)

    print(response)
    # st.write("Reply: ", response["output_text"])
    return response["output_text"]

def reset_conversation():
    st.session_state["prompt"].clear()

def main():
    st.set_page_config("Chat PDF")
    st.header("Legal Insight")

    prompt = st.session_state.get("prompt", [{"role": "system", "content": "none"}])

    st.session_state["prompt"] = prompt
    for message in prompt:
        if message["role"] != "system":
            with st.chat_message(message["role"]):
                st.write(message["content"])

    user_question1 = st.file_uploader("Upload your Input PDF files", accept_multiple_files=True)

    raw_text = get_pdf_text(user_question1)

    user_question = st.chat_input("Ask a Question from the PDF Files")
    





    # if st.button("Submit & Process"):
    #     with st.spinner("Processing..."):
            
    result = ""
    if user_question:
        result = ""
        input = ""
        input = raw_text + "\n" + user_question
        prompt.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.write(user_question)
        with st.chat_message("assistant"):
            botmsg = st.empty()
        result1 = user_input(input)
        # st.write("Reply: ", result1)
        botmsg.write(result1)
        result += result1
        prompt.append({"role": "assistant", "content": result})

        st.session_state["prompt"] = prompt
    st.button('Reset All Chat', on_click=reset_conversation)

    
if __name__ == "__main__":
    main()
