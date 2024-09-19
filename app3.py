import os
import PyPDF2
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from dotenv import load_dotenv
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore

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

# Function to extract text and metadata from PDF
def extract_text_and_metadata_from_pdf(pdf_path):
    all_text = ""
    metadata = []
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        number_of_pages = len(reader.pages)
        for page_num in range(number_of_pages):
            page = reader.pages[page_num]
            page_text = page.extract_text()
            all_text += page_text
            # Extract metadata from the page (e.g., subject, author)
            page_metadata = {
                "subject": page.get('/Subject', 'No subject'),
                "author": page.get('/Author', 'No author')
            }
            metadata.append(page_metadata)
    return all_text, metadata

# Function to format the retrieved documents into a single string
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Streamlit app
def main():
    st.title("PDF Q&A Chatbot")

    # Upload a PDF
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

    if uploaded_file:
        # Process the PDF and split into chunks only once
        with st.spinner('Processing PDF...'):
            pdf_path = "temp_uploaded_pdf.pdf"
            with open(pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
           
            # Extract text and metadata from the PDF
            all_text, metadata = extract_text_and_metadata_from_pdf(pdf_path)
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
                separators=['\n', '\n\n', ' ', '']
            )
            chunks = text_splitter.split_text(text=all_text)

            # Associate metadata with chunks
            chunk_with_metadata = []
            metadata_index = 0
            for chunk in chunks:
                if metadata_index < len(metadata):
                    chunk_with_metadata.append((chunk, metadata[metadata_index]))
                else:
                    chunk_with_metadata.append((chunk, {"subject": "No subject", "author": "No author"}))
                metadata_index += 1

            # Initialize HuggingFace Embeddings & FAISS once
            embeddings = AzureOpenAIEmbeddings(
                model="text-embedding-3-large",
                deployment="TextEmbeddingLarge",
                api_version=azure_api_version,
                azure_endpoint=azure_endpoint,
                openai_api_key=azure_api_key
            )
            
            # Create a vector store with metadata
            vectorstore = FAISS.from_texts([c[0] for c in chunk_with_metadata], embedding=embeddings)
            
            # Initialize InMemoryDocstore to store documents with metadata
            store = InMemoryDocstore()
            for i, (chunk, meta) in enumerate(chunk_with_metadata):
                store.add_document(doc_id=str(i), text=chunk, metadata=meta)
            
            # Initialize ParentDocumentRetriever
            child_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=100,
                length_function=len,
                separators=['\n', '\n\n', ' ', '']
            )
            parent_splitter = RecursiveCharacterTextSplitter(
                chunk_size=2000,
                chunk_overlap=500,
                length_function=len,
                separators=['\n', '\n\n', ' ', '']
            )
            retriever = ParentDocumentRetriever(
                vectorstore=vectorstore,
                docstore=store,
                child_splitter=child_splitter,
                parent_splitter=parent_splitter,
            )

        st.success('PDF processed. You can now ask questions!')
 
        # Azure OpenAI and prompt settings
        llm = AzureChatOpenAI(deployment_name="gpt-4o-mini")
        prompt_template = """Answer the question as precise as possible using the provided context. If the answer is
                            not contained in the context, say "answer not available in context" \n\n
                            Context: \n {context}?\n
                            Question: \n {question} \n
                            Answer:"""
        prompt = PromptTemplate.from_template(template=prompt_template)
 
        # Q&A Interaction
        user_input = st.text_input("Ask your question:")
        if st.button("Submit") and user_input:
            # Retrieve relevant chunks
            with st.spinner('Fetching the answer...'):
                retrieved_docs = retriever.invoke(user_input)
                context = format_docs(retrieved_docs)
               
                # Prepare the input for AzureChatOpenAI
                prompt_input = prompt.format(context=context, question=user_input)
               
                # Get response from the LLM
                response = llm.invoke(prompt_input)
                st.write(f"**Answer:** {response.content}")

if __name__ == "__main__":
    main()
