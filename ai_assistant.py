# logic.py
from langchain_groq.chat_models import ChatGroq
from langchain.chains import RetrievalQA, LLMChain
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from fpdf import FPDF
import fitz
import os

GROQ_API_KEY = "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"
PDF_PATH = "C:\\Users\\mayan\\OneDrive\\Desktop\\Assistant\\CSE332.pdf"
EMBED_MODEL = "nomic-embed-text"

llm_shared = ChatGroq(api_key=GROQ_API_KEY, model="llama3-8b-8192")

custom_pdf_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a helpful assistant. Use ONLY the information below to answer the question.

PDF Context:
{context}

Question: {question}

Answer:"""
)

fallback_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""Answer the question using the following full PDF text as context:

Context:
{context}

Question: {question}

Answer:"""
)

def extract_text_from_pdf(path):
    doc = fitz.open(path)
    return " ".join([page.get_text() for page in doc])

def split_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return splitter.create_documents([text])

def get_vectorstore(pdf_path):
    text = extract_text_from_pdf(pdf_path)
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)

    if os.path.exists("faiss_index/index.faiss"):
        vstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    else:
        chunks = split_text(text)
        vstore = FAISS.from_documents(chunks, embeddings)
        vstore.save_local("faiss_index")
    return vstore, text

def get_chains(vstore, full_text):
    retriever = vstore.as_retriever()
    pdf_qa = RetrievalQA.from_chain_type(
        llm=llm_shared,
        retriever=retriever,
        return_source_documents=False,
        chain_type_kwargs={"prompt": custom_pdf_prompt}
    )
    fallback = LLMChain(llm=llm_shared, prompt=fallback_prompt)
    return pdf_qa, fallback, full_text

def answer_query(pdf_qa, fallback_qa, full_text, question):
    vague_phrases = ["no answer found", "not in pdf", "i don't know", "context does not", "no relevant information"]
    try:
        result = pdf_qa.invoke({"query": question})["result"]
    except:
        result = ""
    if not result or any(v in result.lower() for v in vague_phrases):
        try:
            result = fallback_qa.invoke({"context": full_text, "question": question})["text"]
        except Exception as e:
            result = f"❌ Error from fallback: {str(e)}"
    return result

def save_chat_to_pdf(chat_log, filename):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    font_path = "C:\\Windows\\Fonts\\arial.ttf"
    pdf.add_font("Arial", "", font_path)
    pdf.set_font("Arial", size=12)

    for entry in chat_log:
        pdf.multi_cell(0, 10, entry)
        pdf.ln()

    pdf.output(filename)

