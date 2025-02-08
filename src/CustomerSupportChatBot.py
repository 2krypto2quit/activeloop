from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import PreProcessor, EmbeddingRetriever
from haystack.nodes import PromptNode
from haystack.pipelines import Pipeline
from dotenv import load_dotenv
import os

load_dotenv()

document_store = InMemoryDocumentStore(embedding_dim=1536)

preprocessor = PreProcessor(
   clean_empty_lines=True,
   clean_whitespace=True,
   clean_header_footer=True,
   split_by="word",
   split_length=500,
   split_overlap=50
)

retriever = EmbeddingRetriever(
   document_store=document_store,
   embedding_model="text-embedding-3-small",
   api_key=os.getenv("OPENAI_API_KEY")
)

prompt_node = PromptNode(
   model_name_or_path="gpt-3.5-turbo",
   api_key=os.getenv("OPENAI_API_KEY"),
   default_prompt_template="Given the context: {context}, answer: {query}"
)

pipe = Pipeline()
pipe.add_node(component=retriever, name="Retriever", inputs=["Query"])
pipe.add_node(component=prompt_node, name="PromptNode", inputs=["Retriever"])

docs = [
   {"content": "The solar system consists of eight planets.", "meta": {"source": "astronomy"}},
   {"content": "Mars is known as the red planet.", "meta": {"source": "astronomy"}}
]

processed_docs = preprocessor.process(docs)
document_store.delete_documents()
document_store.write_documents(processed_docs)
document_store.update_embeddings(retriever)

results = pipe.run(
   query="What is Mars known as?",
   params={"Retriever": {"top_k": 1}}
)
print(results)