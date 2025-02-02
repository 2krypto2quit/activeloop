from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import SeleniumURLLoader
import pprint

loader = TextLoader("my_file.txt")
documents = loader.load()
print(f"Number of documents loaded: {len(documents)}")

loader = PyPDFLoader("my_pdf.pdf")
pages = loader.load_and_split()
pprint.pp(pages[0].metadata)

urls = [
    "https://www.youtube.com/watch?v=TFa539R09EQ&t=139s",

]

loader = SeleniumURLLoader(urls)
data = loader.load()
print(data)

from langchain_community.document_loaders import TextLoader, PyPDFLoader, WebBaseLoader
import pprint

# TextLoader for text files
loader = TextLoader("my_file.txt")
documents = loader.load()
print(f"Number of documents loaded: {len(documents)}")

# PyPDFLoader for PDF files
loader = PyPDFLoader("my_pdf.pdf")
pages = loader.load_and_split()
pprint.pp(pages[0].metadata)

# WebBaseLoader for web pages
urls = [
    "https://www.youtube.com/watch?v=TFa539R09EQ&t=139s",
]

loader = WebBaseLoader(urls)
data = loader.load()
print(data)

