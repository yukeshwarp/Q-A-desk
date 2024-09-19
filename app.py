import os
import pdfplumber
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
from dotenv import load_dotenv
 
# Load environment variables from .env file
load_dotenv()
 
# Get Azure OpenAI environment variables from .env file
azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_api_type = os.getenv("OPENAI_API_TYPE")
azure_api_version = os.getenv("OPENAI_API_VERSION")
 
# Set environment variables in current environment (if needed)
os.environ["AZURE_OPENAI_API_KEY"] = azure_api_key
os.environ["AZURE_OPENAI_ENDPOINT"] = azure_endpoint
os.environ["OPENAI_API_TYPE"] = azure_api_type
os.environ["AZURE_OPENAI_API_VERSION"] = azure_api_version
 
# Function to extract text from PDF using pdfplumber
@st.cache_data
def extract_text_from_pdf(pdf_path):
    all_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            all_text += page.extract_text() or ""  # Handle cases where text extraction may fail
    return all_text
 
# Function to format the retrieved documents into a single string
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
 
# Streamlit app
def main():
    st.title("PDF Q&A Chatbot")
 
    # Upload a PDF
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
 
    if uploaded_file:
        st.spinner('Processing PDF...')
        pdf_path = "temp_uploaded_pdf.pdf"
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
       
        # Extract text from the PDF
        all_text = extract_text_from_pdf(pdf_path)
        st.write("Text extraction complete. Length of text:", len(all_text))
 
        # Create Document objects for text splitting
        documents = [Document(page_content=all_text)]
 
        # Initialize AzureChatOpenAI client
        client = AzureChatOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )
 
        # Initialize AzureOpenAIEmbeddings
        embeddings = AzureOpenAIEmbeddings(
            openai_api_version="2024-04-01-preview",
            openai_api_type='azure',
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            azure_deployment='TextEmbeddingLarge'
        )
 
        # Initialize text splitter
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
 
        # Split the extracted PDF data into chunks
        splits = text_splitter.split_documents(documents)
 
        # Add unique IDs to each split
        for i, doc in enumerate(splits):
            doc.metadata = {"id": f"doc_{i}"}
 
        # Compute embeddings for the chunks
        embeddings_list = embeddings.embed_documents([doc.page_content for doc in splits])
 
        # Initialize FAISS
        vectorstore = FAISS(
            embedding_function=embeddings.embed_documents,
            index=None,  # Can initialize an empty FAISS index
            docstore=None,  # Document store initialization can be added as needed
            index_to_docstore_id={i: doc.metadata['id'] for i, doc in enumerate(splits)}
        )
 
        # Add documents to FAISS
        vectorstore.add_documents(splits, embeddings=embeddings_list)
 
        # Set up retriever from FAISS
        retriever = vectorstore.as_retriever()
 
        st.success('PDF processed. You can now ask questions!')
 
        # Define prompt template for the LLM
        prompt_template = """
        Answer the question as accurately as possible using the provided context.
        If the answer is not contained in the context, say "answer not available in context".
 
        Context:
        {context}
 
        Question:
        {question}
 
        Answer:
        """
       
        prompt = PromptTemplate.from_template(template=prompt_template)
 
        # User interaction for Q&A
        user_input = st.text_input("Ask your question:")
       
        if st.button("Submit") and user_input:
            with st.spinner('Fetching the answer...'):
                relevant_docs = retriever.invoke(user_input)
                context = format_docs(relevant_docs)
               
                # Prepare the input for AzureChatOpenAI
                prompt_input = prompt.format(context=context, question=user_input)
               
                # Get response from the LLM
                response = client.invoke({"role": "user", "content": prompt_input})
               
                # Display the response
                st.write(f"**Answer:** {response['content']}")
 
if __name__ == "__main__":
    main()
