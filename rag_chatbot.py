import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import pipeline

load_dotenv()

# 1. Load PDF
pdf_path = "data/sample.pdf"  # Update to your PDF
loader = PyPDFLoader(pdf_path)
documents = loader.load()
print(f"✅ Loaded {len(documents)} pages from PDF.")

# 2. Split into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)
print(f"✅ Split into {len(texts)} chunks.")

# 3. Create embeddings and vector store (offline)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(texts, embeddings, persist_directory="./chroma_db")
print("✅ Created vector store and saved to ./chroma_db")

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 4. Load a small generative model from Hugging Face
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
print(f"⏳ Loading model: {model_name} (first time will download)...")
generator = pipeline("text-generation", model=model_name, device_map="auto")

# 5. Define answer function
def ask_question(query):
    # Retrieve relevant documents
    docs = retriever.get_relevant_documents(query)
    context = "\n\n".join([doc.page_content for doc in docs])

    # Create a prompt with context
    prompt = f"Context: {context}\n\nBased only on the context, answer the following question concisely.\nQuestion: {query}\nAnswer:"

    # Generate answer
    result = generator(prompt, max_new_tokens=150, do_sample=False)[0]['generated_text']
    # Extract the answer part (remove the prompt)
    answer = result[len(prompt):].strip()
    return answer

# 6. Interactive loop
if __name__ == "__main__":
    print("🤖 RAG Chatbot Started (using Hugging Face model)!")
    while True:
        query = input("\n📝 Your question: ")
        if query.lower() in ['quit', 'exit']:
            break
        if query.strip() == "":
            continue
        answer = ask_question(query)
        print(f"\n✅ Answer:\n{answer}\n")