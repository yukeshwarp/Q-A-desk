import os
import PyPDF2
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from dotenv import load_dotenv
 
# Load environment variables from .env file
load_dotenv()
 
# Get Azure OpenAI environment variables from .env file
azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_api_type = os.getenv("OPENAI_API_TYPE")
azure_api_version = os.getenv("OPENAI_API_VERSION")
 
# Set the environment variables
os.environ["AZURE_OPENAI_API_KEY"] = azure_api_key
os.environ["AZURE_OPENAI_ENDPOINT"] = azure_endpoint
os.environ["OPENAI_API_TYPE"] = azure_api_type
os.environ["OPENAI_API_VERSION"] = azure_api_version
 
# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    all_text = ""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        number_of_pages = len(reader.pages)
        for page_num in range(number_of_pages):
            page = reader.pages[page_num]
            page_text = page.extract_text()
            all_text += page_text
    return all_text
 
# Function to format the retrieved documents into a single string
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
 
# Streamlit app
def main():
    st.title("AskPDF - Document explorer")
    st.write("Your document exploration partner.")
    # Upload a PDF
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
 
    if uploaded_file:
        # Process the PDF and split into chunks only once
        with st.spinner('Processing ...'):
            pdf_path = "temp_uploaded_pdf.pdf"
            with open(pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
           
            # Extract text from the PDF
            all_text = extract_text_from_pdf(pdf_path)
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=2000,
                chunk_overlap=500,
                length_function=len,
                separators=['\n', '\n\n', ' ', '']
            )
            chunks = text_splitter.split_text(text=all_text)
 
            # Initialize HuggingFace Embeddings & FAISS once
            embeddings = AzureOpenAIEmbeddings(
        model="text-embedding-3-large",
        deployment="TextEmbeddingLarge",
        api_version=azure_api_version,
        azure_endpoint=azure_endpoint,
        openai_api_key=azure_api_key
    )
            vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
            retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})
 
        st.success('Your document is ready to be explored, post your questions!')
 
        # Azure OpenAI and prompt settings
        llm = AzureChatOpenAI(deployment_name="gpt-4o-mini")
        prompt_template = """Using the provided context, answer the question in a detailed and comprehensive manner. 
                             Make sure your response fully addresses the question and provides as much relevant information as possible. 
                             If the answer is not available in the context, respond with 'Answer not available in context.' 
                             Do not shorten or omit any important details from the answer.

                             Context:
                             {context}

                             Question:
                             {question}

                             Answer:"""
        prompt = PromptTemplate.from_template(template=prompt_template)
 
        # Q&A Interaction
        user_input = st.text_input("Ask your question:")
        if st.button("Ask") and user_input:
            # Retrieve relevant chunks
            with st.spinner('Fetching the answer...'):
                relevant_docs = retriever.invoke(user_input)
                context = format_docs(relevant_docs)
               
                # Prepare the input for AzureChatOpenAI
                prompt_input = prompt.format(context=context, question=user_input)
               
                # Get response from the LLM
                response = llm.invoke(prompt_input)
                st.write(f"**Here's what I've found:** {response.content}")
 
if __name__ == "__main__":
    main()
