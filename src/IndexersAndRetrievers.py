from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import DeepLake
import os
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI  # Changed from OpenAI to ChatOpenAI
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

# Load environment variables
load_dotenv()

# Set Activeloop API key
os.environ["ACTIVELOOP_TOKEN"] = os.getenv("ACTIVELOOP_TOKEN")

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

# Sample text
text = """Google opens up its AI language model PaLM to challenge OpenAI and GPT-3
Google is offering developers access to one of its most advanced AI language models: PaLM.
The search giant is launching an API for PaLM alongside a number of AI enterprise tools
it says will help businesses "generate text, images, code, videos, audio, and more from
simple natural language prompts."

PaLM is a large language model, or LLM, similar to the GPT series created by OpenAI or
Meta's LLaMA family of models. Google first announced PaLM in April 2022. Like other LLMs,
PaLM is a flexible system that can potentially carry out all sorts of text generation and
editing tasks. You could train PaLM to be a conversational chatbot like ChatGPT, for
example, or you could use it for tasks like summarizing text or even writing code.
(It's similar to features Google also announced today for its Workspace apps like Google
Docs and Gmail.)
"""

# Write text to a file
with open("my_file.txt", "w") as f:
    f.write(text)

# Load the text file
loader = TextLoader("my_file.txt")
docs_from_file = loader.load()

# Print the number of documents loaded
print(f"Number of documents loaded: {len(docs_from_file)}")

# Split documents into chunks
text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20)
docs = text_splitter.split_documents(docs_from_file)

# Print the number of chunks created
print(f"Number of document chunks: {len(docs)}")

# Define DeepLake dataset path
active_loop_org_id = "kmatth007"  # Replace with your org ID
active_loop_dataset_name = "langchain_course_indexers_and_retrievers"
dataset_path = f"hub://{active_loop_org_id}/{active_loop_dataset_name}"

# Initialize DeepLake vector store
db = DeepLake(dataset_path=dataset_path, embedding_function=embeddings)

# Add documents to the vector store
db.add_documents(docs)
print("Documents added successfully!")

# Create retriever
retriever = db.as_retriever()

# Create QA chain with ChatOpenAI
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-4"),  # Changed to ChatOpenAI
    retriever=retriever,
    chain_type="stuff"
)

# Run query
query = "How does Google plan to challenge OpenAI?"

# Using invoke instead of run (to avoid deprecation warning)
response = qa_chain.invoke(query)
print(response)

llm = ChatOpenAI(model="gpt-4", temperature=0)
compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor = compressor, base_retriever = retriever)

retrived_docs = compression_retriever.get_relevant_documents("How does Google plan to challenge OpenAI?")

print(retrived_docs[0].page_content)