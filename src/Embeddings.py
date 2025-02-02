from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import DeepLake
import os
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI  # Changed from OpenAI to ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter 

texts = [
    "Napolean Bonaparte was born in 15 August 1759",
    "Louis XIV was born in 5 September 1638",
    "Lady Gaga was born in 28 March 1986",
    "Michael Jeffery Jordan was born in 17 February 1963"
]

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=0,
)

docs = text_splitter.create_documents(texts)

embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

# Load environment variables
load_dotenv()

# Set Activeloop API key
os.environ["ACTIVELOOP_TOKEN"] = os.getenv("ACTIVELOOP_TOKEN")

# Define DeepLake dataset path
active_loop_org_id = "kmatth007"  # Replace with your org ID
active_loop_dataset_name = "langchain_course_embeddings"
dataset_path = f"hub://{active_loop_org_id}/{active_loop_dataset_name}"

# Initialize DeepLake vector store
db = DeepLake(dataset_path=dataset_path, embedding_function=embeddings)

db.add_documents(docs)

retriever = db.as_retriever()

# Create QA chain with ChatOpenAI
qa_chain = RetrievalQA.from_llm(
    llm=ChatOpenAI(model="gpt-4"),  # Changed to ChatOpenAI
    retriever=retriever
)

# Run query
query = "When was Michael Jordon born?"

# Using invoke instead of run (to avoid deprecation warning)
response = qa_chain.invoke(query)
print(response)