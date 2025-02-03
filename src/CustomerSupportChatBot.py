import ssl
import nltk
from langchain_community.document_loaders import SeleniumURLLoader
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import os
import certifi
import requests
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger')

template = """You are an exceptional customer support chatbot that gently answers questions.
You know the following context information:

{chunks_formatted}

Answer the following question from a customer. Use only information from the previous
context information. Do not invent stuff.

Question: {query}

Answer:
"""

prompt = PromptTemplate( 
    input_variables=["chunks_formatted", "query"],
    template=template
)

def fix_ssl_certificates():
    """Setup SSL certificates for Python on macOS"""
    try:
        ssl._create_default_https_context = ssl.create_default_context
        ssl._create_default_https_context().load_verify_locations(certifi.where())
        requests.get('https://www.nltk.org')
        print("SSL certificate fix applied successfully")
    except Exception as e:
        print(f"Error setting up SSL certificates: {str(e)}")

def setup_nltk():
    """Setup NLTK with automatic download of required data"""
    try:
        nltk.download('punkt', quiet=True)
        print("Successfully downloaded NLTK data")
    except Exception as e:
        print(f"Error downloading NLTK data: {str(e)}")
        raise

def load_and_process_urls(urls):
    """Load and process URLs using Selenium with basic configuration"""
    try:
        print("Attempting to load documents...")
        loader = SeleniumURLLoader(urls=urls, continue_on_failure=True)
        docs_not_splitted = loader.load()
        if not docs_not_splitted:
            raise ValueError("No documents were successfully loaded")
        print(f"Successfully loaded {len(docs_not_splitted)} documents")
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs = text_splitter.split_documents(docs_not_splitted)
        print(f"Split into {len(docs)} chunks")
        return docs
    except Exception as e:
        print(f"Error in document loading: {str(e)}")
        return None

def initialize_chroma(persist_directory="chroma_db"):
    """Initialize ChromaDB with a local persistence directory"""
    try:
        # Create embeddings instance
        embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
        
        # Initialize ChromaDB with persistence
        db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        
        print("ChromaDB initialized successfully")
        return db
    except Exception as e:
        print(f"Error initializing ChromaDB: {str(e)}")
        return None

def main():
    try:
        fix_ssl_certificates()
        setup_nltk()
        load_dotenv()

        urls = ['https://beebom.com/what-is-nft-explained/',
                'https://beebom.com/how-delete-spotify-account/',
                'https://beebom.com/how-download-gif-twitter/',
                'https://beebom.com/how-use-chatgpt-linux-terminal/',
                'https://beebom.com/how-delete-spotify-account/',
                'https://beebom.com/how-save-instagram-story-with-music/',
                'https://beebom.com/how-install-pip-windows/',
                'https://beebom.com/how-check-disk-usage-linux/']

        print("Starting document processing...")
        docs = load_and_process_urls(urls)
        if not docs:
            print("Failed to load documents. Exiting.")
            return

        # Setup ChromaDB
        print("Initializing ChromaDB...")
        embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
        
        # Initialize ChromaDB and add documents
        db = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory="chroma_db"
        )
        
        print("Documents added to ChromaDB.")

        print("\nTesting search functionality...")
        query = "How to check disk usage in linux?"
        results = db.similarity_search(query)
        
        # Format the retrieved chunks
        retrieved_chunks = [doc.page_content for doc in results]
        chunks_formatted = "\n\n".join(retrieved_chunks)
        
        # Create OpenAI client
        client = OpenAI()
        
        # Generate answer using OpenAI
        response = client.completions.create(
            model="gpt-3.5-turbo-instruct",
            prompt=prompt.format(
                chunks_formatted=chunks_formatted,
                query=query
            ),
            temperature=0,
            max_tokens=500
        )
        
        if response.choices:
            print(response.choices[0].text.strip())

    except Exception as e:
        print(f"An error occurred in main: {str(e)}")

if __name__ == "__main__":
    main()